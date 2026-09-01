import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BERT.BERT_encoder import load_bert
import math
from einops import rearrange
from .Transformer_mm_mask import WanAttentionSingleStreamBlock, WanAttentionDoubleStreamBlock, rope_params
from .transport import create_transport, Sampler
import random

class DiT(nn.Module):
    def __init__(self, input_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0, clip_dim=512,
                 diff_model='Flow', cond_mask_prob=0.1, num_frame_per_block=3,
                 is_causal=True,
                 **kwargs):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout
        self.num_heads = num_heads
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_frame_per_block = num_frame_per_block
        
        self.cond_mode = cond_mode
        self.cond_mask_prob = cond_mask_prob
        self.is_causal = is_causal
        if self.is_causal:
            print("Using Causal Transformer")
        else:
            print("Using Non-Causal Transformer")

        # InterCMDM
        print('Loading InterCMDM...')
        self.t_embedder = TimestepEmbedder(self.latent_dim)

        # Patchification
        self.x_embedder = nn.Linear(self.input_dim, self.latent_dim)

        print("DiT causal init")
        print ("latent_dim: ", self.latent_dim)
        print ("ff_size: ", self.ff_size)
        print ("num_heads: ", self.num_heads)
        print ("num_layers: ", self.num_layers)
        print ("num_frame_per_block: ", self.num_frame_per_block)
        
        max_length = kwargs.get('max_length', 1024)
        self.freqs = rope_params(max_length, self.latent_dim // self.num_heads)
        print ("max_length: ", max_length)
        assert self.num_layers % 2 == 0, "num_layers must be even"
        self.double_blocks = nn.ModuleList([
            WanAttentionDoubleStreamBlock(self.latent_dim, self.ff_size, self.num_heads, qk_norm=True, cross_attn_norm=True, eps=1e-6)
            for _ in range(self.num_layers)
        ])

        if self.cond_mode == 'text':
            self.y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.final_layer = nn.Linear(self.latent_dim, self.input_dim)
        self.initialize_weights()

        if self.cond_mode == 'text':
            print('Loading CLIP...')
            self.clip_model = self.load_and_freeze_clip()

        self.chunk_size = kwargs.get('chunk_size', 0)
        self.n_tokens = kwargs.get('n_tokens', 75)
        self.clip_noise = kwargs.get('clip_noise', 20.0)
        self.uncertainty_scale = kwargs.get('uncertainty_scale', 1)
        self.sampling_timesteps = kwargs.get('sampling_timesteps', 50)
        self.scheduling_matrix_type = kwargs.get('scheduling_matrix_type', "pyramid")

        print("scheduling_matrix_type: ", self.scheduling_matrix_type)
        print("uncertainty_scale: ", self.uncertainty_scale)
        print("sampling_timesteps: ", self.sampling_timesteps)
        print("chunk_size: ", self.chunk_size)
        print("n_tokens: ", self.n_tokens)

        self.data_max_length = kwargs.get('data_max_length', 75)

        self.same_length_noise = kwargs.get('same_length_noise', False)
        if self.same_length_noise:
            print("Using Same Length Noise for both persons during training")

        self.train_diffusion = create_transport(num_frame_per_block=num_frame_per_block, is_causal=self.is_causal)
        self.gen_diffusion = Sampler(self.train_diffusion)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in DiT blocks:
        for block in self.double_blocks:
            nn.init.constant_(block.adaLN_modulation_p1[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_p1[-1].bias, 0)
            nn.init.constant_(block.adaLN_modulation_p2[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation_p2[-1].bias, 0)

        nn.init.constant_(self.final_layer.weight, 0)
        nn.init.constant_(self.final_layer.bias, 0)


    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self):
        bert_model_path = 'distilbert/distilbert-base-uncased'
        clip_model = load_bert(bert_model_path)
        return clip_model

    def mask_cond(self, cond, force_mask=False):
        t, bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(1, bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, raw_text):
        enc_text, mask = self.clip_model(raw_text)  # self.clip_model.get_last_hidden_state(raw_text, return_mask=True)  # mask: False means no token there
        enc_text = enc_text.permute(1, 0, 2)
        return enc_text, mask

    def forward_step(self, x, t, conds, conds_mask, attention_mask, force_mask=False):
        t = self.t_embedder(t, dtype=x.dtype)
        conds = self.mask_cond(conds, force_mask=force_mask)
        x = self.x_embedder(x)
        x = x.flatten(2)

        if self.training:
            # Multi-task training with all attention mask types
            task_distribution = {
                "simultaneous": 0.60,      # 60% simultaneous interaction (baseline)
                "independent": 0.10,       # 10% independent per-person generation
                "reaction": 0.20,          # 20% reaction generation
                "leader_follower": 0.10,   # 10% following behavior
            }

            attention_mask_type = random.choices(
                list(task_distribution.keys()),
                weights=list(task_distribution.values())
            )[0]

            # Set task-specific parameters
            task_params = {}
            if attention_mask_type == "reaction":
                task_params["reactor_first"] = random.choice([True, False])
            elif attention_mask_type == "leader_follower":
                task_params["lag_frames"] = random.randint(10, 30)
        else:
            # Inference: Use simultaneous as default
            attention_mask_type = getattr(self, 'inference_task', 'simultaneous')
            task_params = getattr(self, 'inference_task_params', {})

        conds = self.y_embedder(conds)
        conds = rearrange(conds, 'len batch dim -> batch len dim', batch=x.shape[0])
        seq_lens = attention_mask.sum(dim=1)
        conds_lens = conds_mask.sum(dim=1)
        self.freqs = self.freqs.to(x.device)

        for block in self.double_blocks:
                x = block(x, t, seq_lens=seq_lens, freqs=self.freqs,
                    context=conds, context_lens=conds_lens, causal=self.is_causal,
                    num_frame_per_block=self.num_frame_per_block,
                    attention_mask_type=attention_mask_type,
                    task_params=task_params)

        x = rearrange(x, 'batch len dim -> batch len 1 dim')
        x = self.final_layer(x)
        return x

    def forward_loss(self, latents, y, m_lens):
        b, l, j, d = latents.shape
        device = latents.device

        max_len = l // 2

        non_pad_mask = lengths_to_mask(m_lens, max_len)
        non_pad_mask_double = non_pad_mask.repeat(1, 2)
        latents = torch.where(non_pad_mask_double.unsqueeze(-1).unsqueeze(-1), latents, torch.zeros_like(latents))

        target = latents.clone()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector, conds_mask = self.encode_text(y)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        attention_mask = non_pad_mask

        model_kwargs = dict(conds=cond_vector, force_mask=force_mask, attention_mask=attention_mask, conds_mask=conds_mask)
        loss_dict = self.train_diffusion.training_losses(self.forward_step, target, model_kwargs, dim=(2, 3), same_length_noise=self.same_length_noise)
        loss = loss_dict["loss"]
        loss = (loss * non_pad_mask_double).sum() / non_pad_mask_double.sum()

        return loss
    

    def forward_with_CFG(self, x, t, conds, conds_mask, attention_mask, cfg=1.0):
        if not cfg == 1.0:
            half = x[: len(x) // 2]
            x = torch.cat([half, half], dim=0)
        x = self.forward_step(x, t, conds, conds_mask, attention_mask)
        if not cfg == 1.0:
            cond_eps, uncond_eps = torch.split(x, len(x) // 2, dim=0)
            half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
            x = torch.cat([half_eps, half_eps], dim=0)
        return x

    def sample(self, x, cond_vector, conds_mask, padding_mask, from_noise_levels, to_noise_levels, cfg=3.0):
        from_noise_levels = rearrange(from_noise_levels, 'len batch -> batch len')
        to_noise_levels = rearrange(to_noise_levels, 'len batch -> batch len')
        pred_v = self.forward_with_CFG(x, from_noise_levels, cond_vector, conds_mask, padding_mask, cfg=cfg)
        delta = to_noise_levels - from_noise_levels
        pred_x = x + delta.unsqueeze(-1).unsqueeze(-1) * pred_v
        return pred_x

    def generate(self, conds, m_lens, cond_scale=1.0, verbose=False):
        device = next(self.parameters()).device

        b = len(m_lens)
        l = max(m_lens)

        non_pad_mask = lengths_to_mask(m_lens, max(m_lens))

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector, conds_mask = self.encode_text(conds)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=1)
            conds_mask = torch.cat([conds_mask, conds_mask], dim=0)
            non_pad_mask = torch.cat([non_pad_mask, non_pad_mask], dim=0)

        xs_pred_1 = torch.zeros((b, 0, 1, self.input_dim), device=device)
        xs_pred_2 = torch.zeros((b, 0, 1, self.input_dim), device=device)


        curr_frame = 0
        n_frames = l
        batch_size = b
        self.device = device

        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            horizon_block = max(1, horizon // self.num_frame_per_block)
            if horizon % self.num_frame_per_block != 0:
                horizon_block += 1
            scheduling_matrix = self._generate_scheduling_matrix(horizon_block)
            scheduling_matrix = scheduling_matrix.repeat(self.num_frame_per_block, axis=1)
            scheduling_matrix = scheduling_matrix[:, :horizon]
            scheduling_matrix = 1 - scheduling_matrix / self.sampling_timesteps

            chunk_1 = torch.randn((batch_size, horizon, 1, self.input_dim), device=self.device)
            chunk_2 = torch.randn((batch_size, horizon, 1, self.input_dim), device=self.device)

            chunk_1 = torch.clamp(chunk_1, -self.clip_noise, self.clip_noise)
            chunk_2 = torch.clamp(chunk_2, -self.clip_noise, self.clip_noise)

            xs_pred_1 = torch.cat([xs_pred_1, chunk_1], dim=1)
            xs_pred_2 = torch.cat([xs_pred_2, chunk_2], dim=1)

            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            if verbose:
                print (f"start_frame: {start_frame} | curr_frame: {curr_frame} | horizon: {horizon} | window: {curr_frame + horizon - start_frame}")

            for m in range(scheduling_matrix.shape[0] - 1):
                from_noise_levels = np.concatenate((np.ones((curr_frame,), dtype=np.int64), scheduling_matrix[m]))[
                    :, None
                ].repeat(batch_size, axis=1)
                to_noise_levels = np.concatenate(
                    (
                        np.ones((curr_frame,), dtype=np.int64),
                        scheduling_matrix[m + 1],
                    )
                )[
                    :, None
                ].repeat(batch_size, axis=1)

                from_noise_levels = torch.from_numpy(from_noise_levels).to(self.device).float()
                to_noise_levels = torch.from_numpy(to_noise_levels).to(self.device).float()

                max_length = self.data_max_length
                now_length = xs_pred_1[:, start_frame:].shape[1]
                if now_length < max_length:
                    padding_length = max_length - now_length
                    padding_from_noise_levels = torch.zeros((padding_length, batch_size), device=self.device).float()
                    padding_to_noise_levels = torch.zeros((padding_length, batch_size), device=self.device).float()
                    from_noise_levels_pad = torch.cat([from_noise_levels[start_frame:, :], padding_from_noise_levels], dim=0)
                    to_noise_levels_pad = torch.cat([to_noise_levels[start_frame:, :], padding_to_noise_levels], dim=0)
                    padding_xs_pred_1 = torch.zeros((batch_size, padding_length, 1, self.input_dim), device=self.device).float()
                    padding_xs_pred_2 = torch.zeros((batch_size, padding_length, 1, self.input_dim), device=self.device).float()
                    xs_pred_1_pad = torch.cat([xs_pred_1[:, start_frame:], padding_xs_pred_1], dim=1)
                    xs_pred_2_pad = torch.cat([xs_pred_2[:, start_frame:], padding_xs_pred_2], dim=1)

                    from_noise_levels = torch.cat([from_noise_levels_pad, from_noise_levels_pad], dim=0)
                    to_noise_levels = torch.cat([to_noise_levels_pad, to_noise_levels_pad], dim=0)
                    xs_pred_input = torch.cat([xs_pred_1_pad, xs_pred_2_pad], dim=1)

                    non_pad_mask_input = non_pad_mask[:, start_frame : curr_frame + horizon]

                elif now_length == max_length:
                    xs_pred_1_pad = xs_pred_1[:, start_frame:]
                    xs_pred_2_pad = xs_pred_2[:, start_frame:]

                    from_noise_levels = torch.cat([from_noise_levels[start_frame:, :], from_noise_levels[start_frame:, :]], dim=0)
                    to_noise_levels = torch.cat([to_noise_levels[start_frame:, :], to_noise_levels[start_frame:, :]], dim=0)
                    xs_pred_input = torch.cat([xs_pred_1_pad, xs_pred_2_pad], dim=1)
                    non_pad_mask_input = non_pad_mask[:, start_frame : curr_frame + horizon]

                else:
                    raise ValueError("windows length is larger than the maximum number of tokens")

                if not cond_scale == 1.0:
                    xs_pred_input = torch.cat([xs_pred_input, xs_pred_input], dim=0)
                    from_noise_levels = torch.cat([from_noise_levels, from_noise_levels], dim=1)
                    to_noise_levels = torch.cat([to_noise_levels, to_noise_levels], dim=1)
                
                generated_latents = self.sample(
                    xs_pred_input,
                    cond_vector=cond_vector,
                    conds_mask=conds_mask,
                    padding_mask=non_pad_mask_input,
                    from_noise_levels=from_noise_levels,
                    to_noise_levels=to_noise_levels,
                    cfg=cond_scale
                )

                if not cond_scale == 1.0:
                    generated_latents = generated_latents.chunk(2, dim=0)[0]

                xs_pred_1[:, start_frame:] = generated_latents[:, :now_length]
                xs_pred_2[:, start_frame:] = generated_latents[:, max_length:max_length+now_length]

            curr_frame += horizon

        return xs_pred_1, xs_pred_2

    def generate_with_gt_person1(self, conds, m_lens, gt_latent_1, cond_scale=1.0, verbose=False):
        """Generate Person 2's motion conditioned on Person 1's ground truth latent.

        Person 1's latent is kept clean (noise level = 1.0) throughout the diffusion
        process so the model can attend to it while denoising only Person 2.

        Args:
            conds: Text conditions.
            m_lens: Latent sequence lengths (motion_lens // 4).
            gt_latent_1: Ground truth latent for Person 1, shape [b, l, 1, input_dim].
            cond_scale: Classifier-free guidance scale.
        """
        device = next(self.parameters()).device

        b = len(m_lens)
        l = max(m_lens)

        non_pad_mask = lengths_to_mask(m_lens, max(m_lens))

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector, conds_mask = self.encode_text(conds)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=1)
            conds_mask = torch.cat([conds_mask, conds_mask], dim=0)
            non_pad_mask = torch.cat([non_pad_mask, non_pad_mask], dim=0)

        xs_pred_1 = gt_latent_1[:, :l].to(device)
        xs_pred_2 = torch.zeros((b, 0, 1, self.input_dim), device=device)

        curr_frame = 0
        n_frames = l
        batch_size = b
        self.device = device

        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            horizon_block = max(1, horizon // self.num_frame_per_block)
            if horizon % self.num_frame_per_block != 0:
                horizon_block += 1
            scheduling_matrix = self._generate_scheduling_matrix(horizon_block)
            scheduling_matrix = scheduling_matrix.repeat(self.num_frame_per_block, axis=1)
            scheduling_matrix = scheduling_matrix[:, :horizon]
            scheduling_matrix = 1 - scheduling_matrix / self.sampling_timesteps

            chunk_2 = torch.randn((batch_size, horizon, 1, self.input_dim), device=self.device)
            chunk_2 = torch.clamp(chunk_2, -self.clip_noise, self.clip_noise)
            xs_pred_2 = torch.cat([xs_pred_2, chunk_2], dim=1)

            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            if verbose:
                print(f"[gt_p1] start_frame: {start_frame} | curr_frame: {curr_frame} | horizon: {horizon}")

            for m in range(scheduling_matrix.shape[0] - 1):
                p2_from_noise = np.concatenate(
                    (np.ones((curr_frame,), dtype=np.int64), scheduling_matrix[m])
                )[:, None].repeat(batch_size, axis=1)
                p2_to_noise = np.concatenate(
                    (np.ones((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1])
                )[:, None].repeat(batch_size, axis=1)

                p2_from_noise = torch.from_numpy(p2_from_noise).to(self.device).float()
                p2_to_noise = torch.from_numpy(p2_to_noise).to(self.device).float()

                max_length = self.data_max_length
                now_length = xs_pred_2[:, start_frame:].shape[1]

                if now_length < max_length:
                    padding_length = max_length - now_length
                    padding_zeros = torch.zeros((batch_size, padding_length, 1, self.input_dim), device=self.device)
                    padding_noise = torch.zeros((padding_length, batch_size), device=self.device)

                    p1_slice = xs_pred_1[:, start_frame:start_frame + now_length]
                    p1_pad_len = max_length - p1_slice.shape[1]
                    if p1_pad_len > 0:
                        p1_slice = torch.cat([p1_slice, torch.zeros((batch_size, p1_pad_len, 1, self.input_dim), device=self.device)], dim=1)
                    p1_from = torch.ones((max_length, batch_size), device=self.device)
                    p1_to = torch.ones((max_length, batch_size), device=self.device)

                    xs_pred_2_pad = torch.cat([xs_pred_2[:, start_frame:], padding_zeros], dim=1)
                    p2_from_pad = torch.cat([p2_from_noise[start_frame:], padding_noise], dim=0)
                    p2_to_pad = torch.cat([p2_to_noise[start_frame:], padding_noise], dim=0)

                    from_noise_levels = torch.cat([p1_from, p2_from_pad], dim=0)
                    to_noise_levels = torch.cat([p1_to, p2_to_pad], dim=0)
                    xs_pred_input = torch.cat([p1_slice, xs_pred_2_pad], dim=1)
                    non_pad_mask_input = non_pad_mask[:, start_frame:curr_frame + horizon]

                elif now_length == max_length:
                    p1_slice = xs_pred_1[:, start_frame:start_frame + max_length]
                    xs_pred_2_pad = xs_pred_2[:, start_frame:]

                    p1_from = torch.ones((max_length, batch_size), device=self.device)
                    p1_to = torch.ones((max_length, batch_size), device=self.device)

                    from_noise_levels = torch.cat([p1_from, p2_from_noise[start_frame:]], dim=0)
                    to_noise_levels = torch.cat([p1_to, p2_to_noise[start_frame:]], dim=0)
                    xs_pred_input = torch.cat([p1_slice, xs_pred_2_pad], dim=1)
                    non_pad_mask_input = non_pad_mask[:, start_frame:curr_frame + horizon]
                else:
                    raise ValueError("window length is larger than the maximum number of tokens")

                if not cond_scale == 1.0:
                    xs_pred_input = torch.cat([xs_pred_input, xs_pred_input], dim=0)
                    from_noise_levels = torch.cat([from_noise_levels, from_noise_levels], dim=1)
                    to_noise_levels = torch.cat([to_noise_levels, to_noise_levels], dim=1)

                generated_latents = self.sample(
                    xs_pred_input,
                    cond_vector=cond_vector,
                    conds_mask=conds_mask,
                    padding_mask=non_pad_mask_input,
                    from_noise_levels=from_noise_levels,
                    to_noise_levels=to_noise_levels,
                    cfg=cond_scale
                )

                if not cond_scale == 1.0:
                    generated_latents = generated_latents.chunk(2, dim=0)[0]

                xs_pred_2[:, start_frame:] = generated_latents[:, max_length:max_length + now_length]

            curr_frame += horizon

        return xs_pred_1, xs_pred_2

    def generate_with_gt_person2(self, conds, m_lens, gt_latent_2, cond_scale=1.0, verbose=False):
        """Generate Person 1's motion conditioned on Person 2's ground truth latent.

        Person 2's latent is kept clean (noise level = 1.0) throughout the diffusion
        process so the model can attend to it while denoising only Person 1.

        Args:
            conds: Text conditions.
            m_lens: Latent sequence lengths (motion_lens // 4).
            gt_latent_2: Ground truth latent for Person 2, shape [b, l, 1, input_dim].
            cond_scale: Classifier-free guidance scale.
        """
        device = next(self.parameters()).device

        b = len(m_lens)
        l = max(m_lens)

        non_pad_mask = lengths_to_mask(m_lens, max(m_lens))

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector, conds_mask = self.encode_text(conds)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=1)
            conds_mask = torch.cat([conds_mask, conds_mask], dim=0)
            non_pad_mask = torch.cat([non_pad_mask, non_pad_mask], dim=0)

        xs_pred_1 = torch.zeros((b, 0, 1, self.input_dim), device=device)
        xs_pred_2 = gt_latent_2[:, :l].to(device)

        curr_frame = 0
        n_frames = l
        batch_size = b
        self.device = device

        while curr_frame < n_frames:
            if self.chunk_size > 0:
                horizon = min(n_frames - curr_frame, self.chunk_size)
            else:
                horizon = n_frames - curr_frame
            assert horizon <= self.n_tokens, "horizon exceeds the number of tokens."
            horizon_block = max(1, horizon // self.num_frame_per_block)
            if horizon % self.num_frame_per_block != 0:
                horizon_block += 1
            scheduling_matrix = self._generate_scheduling_matrix(horizon_block)
            scheduling_matrix = scheduling_matrix.repeat(self.num_frame_per_block, axis=1)
            scheduling_matrix = scheduling_matrix[:, :horizon]
            scheduling_matrix = 1 - scheduling_matrix / self.sampling_timesteps

            chunk_1 = torch.randn((batch_size, horizon, 1, self.input_dim), device=self.device)
            chunk_1 = torch.clamp(chunk_1, -self.clip_noise, self.clip_noise)
            xs_pred_1 = torch.cat([xs_pred_1, chunk_1], dim=1)

            start_frame = max(0, curr_frame + horizon - self.n_tokens)

            if verbose:
                print(f"[gt_p2] start_frame: {start_frame} | curr_frame: {curr_frame} | horizon: {horizon}")

            for m in range(scheduling_matrix.shape[0] - 1):
                p1_from_noise = np.concatenate(
                    (np.ones((curr_frame,), dtype=np.int64), scheduling_matrix[m])
                )[:, None].repeat(batch_size, axis=1)
                p1_to_noise = np.concatenate(
                    (np.ones((curr_frame,), dtype=np.int64), scheduling_matrix[m + 1])
                )[:, None].repeat(batch_size, axis=1)

                p1_from_noise = torch.from_numpy(p1_from_noise).to(self.device).float()
                p1_to_noise = torch.from_numpy(p1_to_noise).to(self.device).float()

                max_length = self.data_max_length
                now_length = xs_pred_1[:, start_frame:].shape[1]

                if now_length < max_length:
                    padding_length = max_length - now_length
                    padding_zeros = torch.zeros((batch_size, padding_length, 1, self.input_dim), device=self.device)
                    padding_noise = torch.zeros((padding_length, batch_size), device=self.device)

                    xs_pred_1_pad = torch.cat([xs_pred_1[:, start_frame:], padding_zeros], dim=1)
                    p1_from_pad = torch.cat([p1_from_noise[start_frame:], padding_noise], dim=0)
                    p1_to_pad = torch.cat([p1_to_noise[start_frame:], padding_noise], dim=0)

                    p2_slice = xs_pred_2[:, start_frame:start_frame + now_length]
                    p2_pad_len = max_length - p2_slice.shape[1]
                    if p2_pad_len > 0:
                        p2_slice = torch.cat([p2_slice, torch.zeros((batch_size, p2_pad_len, 1, self.input_dim), device=self.device)], dim=1)
                    p2_from = torch.ones((max_length, batch_size), device=self.device)
                    p2_to = torch.ones((max_length, batch_size), device=self.device)

                    from_noise_levels = torch.cat([p1_from_pad, p2_from], dim=0)
                    to_noise_levels = torch.cat([p1_to_pad, p2_to], dim=0)
                    xs_pred_input = torch.cat([xs_pred_1_pad, p2_slice], dim=1)
                    non_pad_mask_input = non_pad_mask[:, start_frame:curr_frame + horizon]

                elif now_length == max_length:
                    xs_pred_1_pad = xs_pred_1[:, start_frame:]
                    p2_slice = xs_pred_2[:, start_frame:start_frame + max_length]

                    p2_from = torch.ones((max_length, batch_size), device=self.device)
                    p2_to = torch.ones((max_length, batch_size), device=self.device)

                    from_noise_levels = torch.cat([p1_from_noise[start_frame:], p2_from], dim=0)
                    to_noise_levels = torch.cat([p1_to_noise[start_frame:], p2_to], dim=0)
                    xs_pred_input = torch.cat([xs_pred_1_pad, p2_slice], dim=1)
                    non_pad_mask_input = non_pad_mask[:, start_frame:curr_frame + horizon]
                else:
                    raise ValueError("window length is larger than the maximum number of tokens")

                if not cond_scale == 1.0:
                    xs_pred_input = torch.cat([xs_pred_input, xs_pred_input], dim=0)
                    from_noise_levels = torch.cat([from_noise_levels, from_noise_levels], dim=1)
                    to_noise_levels = torch.cat([to_noise_levels, to_noise_levels], dim=1)

                generated_latents = self.sample(
                    xs_pred_input,
                    cond_vector=cond_vector,
                    conds_mask=conds_mask,
                    padding_mask=non_pad_mask_input,
                    from_noise_levels=from_noise_levels,
                    to_noise_levels=to_noise_levels,
                    cfg=cond_scale
                )

                if not cond_scale == 1.0:
                    generated_latents = generated_latents.chunk(2, dim=0)[0]

                xs_pred_1[:, start_frame:] = generated_latents[:, :now_length]

            curr_frame += horizon

        return xs_pred_1, xs_pred_2

    def _generate_scheduling_matrix(self, horizon: int):
        match self.scheduling_matrix_type:
            case "pyramid":
                return self._generate_pyramid_scheduling_matrix(horizon, self.uncertainty_scale)
            case "full_sequence":
                return np.arange(self.sampling_timesteps, -1, -1)[:, None].repeat(horizon.cpu().numpy() if isinstance(horizon, torch.Tensor) else horizon, axis=1)
            case "autoregressive":
                return self._generate_pyramid_scheduling_matrix(horizon, self.sampling_timesteps)
            case "trapezoid":
                return self._generate_trapezoid_scheduling_matrix(horizon, self.uncertainty_scale)

    def _generate_pyramid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon - 1) * uncertainty_scale) + 1
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range(horizon):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)

    def _generate_trapezoid_scheduling_matrix(self, horizon: int, uncertainty_scale: float):
        height = self.sampling_timesteps + int((horizon + 1) // 2 * uncertainty_scale)
        scheduling_matrix = np.zeros((height, horizon), dtype=np.int64)
        for m in range(height):
            for t in range((horizon + 1) // 2):
                scheduling_matrix[m, t] = self.sampling_timesteps + int(t * uncertainty_scale) - m
                scheduling_matrix[m, -t] = self.sampling_timesteps + int(t * uncertainty_scale) - m

        return np.clip(scheduling_matrix, 0, self.sampling_timesteps)


#################################################################################
#                                     InterCMDM Zoos                                #
#################################################################################
def dit(**kwargs):
    layer = 8
    return DiT(latent_dim=layer*64, ff_size=layer*64*2, num_layers=layer, num_heads=layer//2, dropout=0, clip_dim=768,
                 diff_model="Flow", cond_mask_prob=0.1, **kwargs)

def dit_large(**kwargs):
    layer = 16
    return DiT(latent_dim=layer*64, ff_size=layer*64*2, num_layers=layer, num_heads=layer//2, dropout=0, clip_dim=768,
                 diff_model="Flow", cond_mask_prob=0.1, **kwargs)

def dit_mlarge(**kwargs):
    layer = 12
    return DiT(latent_dim=layer*64, ff_size=layer*64*2, num_layers=layer, num_heads=layer//2, dropout=0, clip_dim=768,
                 diff_model="Flow", cond_mask_prob=0.1, **kwargs)

def dit_light(**kwargs):
    layer = 4
    return DiT(latent_dim=512, ff_size=1024, num_layers=layer, num_heads=layer//2, dropout=0, clip_dim=768,
                 diff_model="Flow", cond_mask_prob=0.1, **kwargs)

#################################################################################
#                                 Inner Architectures                           #
#################################################################################

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        if t.dim() == 1:
            t = t.unsqueeze(1)
        
        B, L = t.shape
        t = t.reshape(B * L)
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        embedding = embedding.reshape(B, L, -1)
        return embedding

    def forward(self, t, dtype=torch.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask #(b, len)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)