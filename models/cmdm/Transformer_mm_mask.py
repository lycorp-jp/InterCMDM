import torch
import torch.nn as nn

import math
from einops import rearrange


# @amp.autocast(enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    # compute in float32 for speed; returns complex64
    inv = torch.arange(0, dim, 2, dtype=torch.float32) / float(dim)
    base = torch.tensor(float(theta), dtype=torch.float32)
    freqs = torch.outer(
        torch.arange(max_seq_len, dtype=torch.float32),
        1.0 / torch.pow(base, inv))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


# @amp.autocast(enabled=False)
def rope_apply(x, freqs):
    n, c = x.size(2), x.size(3) // 2

    # fast path: vectorized apply up to max length, then restore padding per sample
    # cast once to complex for whole batch (use float32/complex64 for speed)
    dtype_orig = x.dtype
    max_len = x.size(1)
    xb = x
    xc = torch.view_as_complex(xb.to(torch.float32).reshape(xb.size(0), max_len, n, -1, 2))  # (B, Fmax, N, C/2)

    # slice freqs to max_len and broadcast over batch/heads
    freqs_i = freqs[:max_len].to(xc.dtype).view(max_len, 1, -1)  # (Fmax, 1, C)

    # apply rotary embedding (time-only) in one go
    yc = xc * freqs_i  # (B, Fmax, N, C/2)
    yr = torch.view_as_real(yc)
    yr = yr.flatten(3)  # (B, Fmax, N, C)
    return yr.to(dtype_orig)


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x).type_as(x)

def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    causal=False,
    num_frame_per_block=1,
    mask_per_person=True,
    dtype=torch.float,
    attention_mask_type=None,
    task_params=None,  # NEW: Task-specific parameters
):
    # Build attention mask from q_lens / k_lens if provided, and combine with causal mask
    b, lq, lk = q.size(0), q.size(1), k.size(1)
    device = q.device
    attn_mask = None
    if not (q_lens is None and k_lens is None):
        # Default to full lengths when lens are not provided
        if q_lens is None:
            q_lens = torch.full((b,), lq, dtype=torch.int32, device=device)
        else:
            q_lens = q_lens.to(device=device, dtype=torch.int32)
        if k_lens is None:
            k_lens = torch.full((b,), lk, dtype=torch.int32, device=device)
        else:
            k_lens = k_lens.to(device=device, dtype=torch.int32)

        
        q_positions = torch.arange(lq, device=device, dtype=torch.int32).unsqueeze(0)  # [1, Lq]
        q_pad = q_positions >= q_lens.unsqueeze(1)  # [B, Lq]

        if mask_per_person:
            assert lq == lk and lq % 2 == 0
            # True indicates positions to be masked (not attended)
            person_k_len = lk // 2 if lk // 2 > 0 else lk
            if person_k_len == 0:
                k_pad = torch.zeros((b, 0), dtype=torch.bool, device=device)
            else:
                k_positions_single = torch.arange(
                    person_k_len, device=device, dtype=torch.int32
                ).unsqueeze(0)  # [1, Lk/2]
                single_pad = k_positions_single >= k_lens.unsqueeze(1)  # mask per person
                k_pad = torch.cat([single_pad, single_pad], dim=1)
                if q_pad.size(1) < lq:
                    extra = single_pad[:, : (lk - k_pad.size(1))]
                    k_pad = torch.cat([k_pad, extra], dim=1)
                elif k_pad.size(1) > lk:
                    k_pad = k_pad[:, :lk]
        else:
            k_positions = torch.arange(lk, device=device, dtype=torch.int32).unsqueeze(0)  # [1, Lk]
            k_pad = k_positions >= k_lens.unsqueeze(1)  # [B, Lk]

        # Broadcast over heads: [B, 1, Lq, Lk]
        attn_mask = q_pad.unsqueeze(1).unsqueeze(3) | k_pad.unsqueeze(1).unsqueeze(1)

    if causal and attention_mask_type != "non_causal":
        assert lq == lk

        def build_causal_mask(lq, num_frame_per_block):
            total_length = lq
            ends = torch.zeros(total_length,
                                device=device, dtype=torch.long)

            # Block-wise causal mask will attend to all elements that are before the end of the current chunk
            frame_indices = torch.arange(
                start=0,
                end=total_length,
                step=num_frame_per_block,
                device=device
            )

            for tmp in frame_indices:
                ends[tmp:tmp + num_frame_per_block] = tmp + num_frame_per_block

            def attention_mask(q_idx, kv_idx):
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)

            q_idx = torch.arange(lq, device=device).unsqueeze(1)
            k_idx = torch.arange(lq, device=device).unsqueeze(0)

            causal_mask = attention_mask(q_idx, k_idx)
            return causal_mask

        # Initialize task_params
        task_params = task_params or {}

        if attention_mask_type == "reaction":
            # Reaction Generation: One person reacts to another's complete motion
            assert lq % 2 == 0 and lq > 0
            reactor_first = task_params.get("reactor_first", False)
            causal_mask = torch.zeros((lq, lq), dtype=torch.bool, device=device)
            frames_per_person = lq // 2
            single_causal_mask = build_causal_mask(frames_per_person, num_frame_per_block)
            full_mask = torch.ones((frames_per_person, frames_per_person), dtype=torch.bool, device=device)

            if reactor_first:
                # Person 1 is the reactor (being generated)
                causal_mask[0:frames_per_person, 0:frames_per_person] = single_causal_mask  # P1 self: causal
                causal_mask[0:frames_per_person, frames_per_person:lq] = True  # P1→P2: FULL (reactor sees all) - FIXED
                causal_mask[frames_per_person:lq, frames_per_person:lq] = full_mask  # P2 self: FULL (observed) - FIXED
                causal_mask[frames_per_person:lq, 0:frames_per_person] = single_causal_mask  # P2→P1: causal
            else:
                # Person 2 is the reactor (being generated) - DEFAULT
                causal_mask[0:frames_per_person, 0:frames_per_person] = full_mask  # P1 self: FULL (observed) - FIXED
                causal_mask[0:frames_per_person, frames_per_person:lq] = single_causal_mask  # P1→P2: causal
                causal_mask[frames_per_person:lq, frames_per_person:lq] = single_causal_mask  # P2 self: causal
                causal_mask[frames_per_person:lq, 0:frames_per_person] = True  # P2→P1: FULL (reactor sees all) - FIXED

        elif attention_mask_type == "leader_follower":
            # Leader-Follower: Person 2 follows Person 1 with temporal lag
            assert lq % 2 == 0 and lq > 0
            lag_frames = task_params.get("lag_frames", 3)
            causal_mask = torch.zeros((lq, lq), dtype=torch.bool, device=device)
            frames_per_person = lq // 2
            single_causal_mask = build_causal_mask(frames_per_person, num_frame_per_block)

            # Person 1 (leader): Standard causal self-attention
            causal_mask[0:frames_per_person, 0:frames_per_person] = single_causal_mask
            # Person 2 (follower): Causal self-attention
            causal_mask[frames_per_person:lq, frames_per_person:lq] = single_causal_mask
            # P1→P2: No attention (leader doesn't need follower) - FIXED
            causal_mask[0:frames_per_person, frames_per_person:lq] = False
            # P2→P1: Local lag window centered around follower timestep i.
            # For lag_frames=2, follower i can attend leader [i-1, i, i+1] (clipped).
            for i in range(frames_per_person):
                start_idx = max(0, i - lag_frames + 1)
                end_idx = min(frames_per_person, i + lag_frames)
                causal_mask[frames_per_person + i, start_idx:end_idx] = True

        elif attention_mask_type == "independent":
            # Independent: Each person generated independently with no cross-attention
            assert lq % 2 == 0 and lq > 0
            causal_mask = torch.zeros((lq, lq), dtype=torch.bool, device=device)
            frames_per_person = lq // 2
            single_causal_mask = build_causal_mask(frames_per_person, num_frame_per_block)
            # Person 1 self-attention is causal
            causal_mask[0:frames_per_person, 0:frames_per_person] = single_causal_mask
            # Person 2 self-attention is causal
            causal_mask[frames_per_person:lq, frames_per_person:lq] = single_causal_mask
            # Person 1 attending to person 2: No attention
            causal_mask[0:frames_per_person, frames_per_person:lq] = False
            # Person 2 attending to person 1: No attention
            causal_mask[frames_per_person:lq, 0:frames_per_person] = False

        elif attention_mask_type == "simultaneous":
            # Simultaneous: Both persons interact with causal cross-attention
            assert lq % 2 == 0 and lq > 0
            causal_mask = torch.zeros((lq, lq), dtype=torch.bool, device=device)
            frames_per_person = lq // 2
            single_causal_mask = build_causal_mask(frames_per_person, num_frame_per_block)
            causal_mask[0:frames_per_person, 0:frames_per_person] = single_causal_mask
            causal_mask[frames_per_person:lq, frames_per_person:lq] = single_causal_mask
            causal_mask[0:frames_per_person, frames_per_person:lq] = single_causal_mask
            causal_mask[frames_per_person:lq, 0:frames_per_person] = single_causal_mask
        else:
            causal_mask = build_causal_mask(lq, num_frame_per_block)

        causal_mask = ~causal_mask.unsqueeze(0).unsqueeze(0)

        attn_mask = causal_mask if attn_mask is None else (attn_mask | causal_mask)
    
    if attn_mask is not None:
        attn_mask = ~attn_mask

    q = q.transpose(1, 2).to(dtype)
    k = k.transpose(1, 2).to(dtype)
    v = v.transpose(1, 2).to(dtype)

    out = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, dropout_p=dropout_p)

    out = out.transpose(1, 2).contiguous()
    return out



class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, freqs=None, causal=False, num_frame_per_block=1):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs (time-only), shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        q_p1, q_p2 = q.chunk(2, dim=1)
        k_p1, k_p2 = k.chunk(2, dim=1)

        q_p1 = rope_apply(q_p1, freqs) if freqs is not None else q_p1
        q_p2 = rope_apply(q_p2, freqs) if freqs is not None else q_p2
        k_p1 = rope_apply(k_p1, freqs) if freqs is not None else k_p1
        k_p2 = rope_apply(k_p2, freqs) if freqs is not None else k_p2

        q = torch.cat((q_p1, q_p2), dim=1)
        k = torch.cat((k_p1, k_p2), dim=1)

        x = attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            causal=causal,
            num_frame_per_block=num_frame_per_block,
            mask_per_person=True)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x

class WanSelfAttentionDoubleStream(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q_p1 = nn.Linear(dim, dim)
        self.q_p2 = nn.Linear(dim, dim)
        self.k_p1 = nn.Linear(dim, dim)
        self.k_p2 = nn.Linear(dim, dim)
        self.v_p1 = nn.Linear(dim, dim)
        self.v_p2 = nn.Linear(dim, dim)
        self.o_p1 = nn.Linear(dim, dim)
        self.o_p2 = nn.Linear(dim, dim)
        self.norm_q_p1 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_q_p2 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        self.norm_k_p1 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k_p2 = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x_p1, x_p2, seq_lens, freqs=None, causal=False, num_frame_per_block=1, attention_mask_type=None, task_params=None):
        r"""
        Args:
            x_p1(Tensor): Shape [B, L, num_heads, C / num_heads]
            x_p2(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            freqs(Tensor): Rope freqs (time-only), shape [1024, C / num_heads / 2]
            attention_mask_type(str): Type of attention mask to use
            task_params(dict): Task-specific parameters
        """
        b, s, n, d = *x_p1.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn_p1(x):
            q = self.norm_q_p1(self.q_p1(x)).view(b, s, n, d)
            k = self.norm_k_p1(self.k_p1(x)).view(b, s, n, d)
            v = self.v_p1(x).view(b, s, n, d)
            return q, k, v

        def qkv_fn_p2(x):
            q = self.norm_q_p2(self.q_p2(x)).view(b, s, n, d)
            k = self.norm_k_p2(self.k_p2(x)).view(b, s, n, d)
            v = self.v_p2(x).view(b, s, n, d)
            return q, k, v

        q_p1, k_p1, v_p1 = qkv_fn_p1(x_p1)  
        q_p2, k_p2, v_p2 = qkv_fn_p2(x_p2)
        q_p1 = rope_apply(q_p1, freqs) if freqs is not None else q_p1
        q_p2 = rope_apply(q_p2, freqs) if freqs is not None else q_p2
        k_p1 = rope_apply(k_p1, freqs) if freqs is not None else k_p1
        k_p2 = rope_apply(k_p2, freqs) if freqs is not None else k_p2
        v_p1 = v_p1
        v_p2 = v_p2

        q = torch.cat((q_p1, q_p2), dim=1)
        k = torch.cat((k_p1, k_p2), dim=1)
        v = torch.cat((v_p1, v_p2), dim=1)
        
        x = attention(
            q=q,
            k=k,
            v=v,
            k_lens=seq_lens,
            causal=causal,
            num_frame_per_block=num_frame_per_block,
            mask_per_person=True,
            attention_mask_type=attention_mask_type,
            task_params=task_params)

        # output
        x = x.flatten(2)
        x_p1, x_p2 = x.chunk(2, dim=1)
        x_p1 = self.o_p1(x_p1)
        x_p2 = self.o_p2(x_p2)

        return x_p1, x_p2


class WanT2VCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens, freqs=None, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        q = rope_apply(q, freqs) if freqs is not None else q

        # compute attention
        x = attention(q, k, v, k_lens=context_lens, mask_per_person=False)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionSingleStreamBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanT2VCrossAttention(dim,
                                                num_heads,
                                                qk_norm,
                                                eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 6 * self.dim, bias=True)
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        freqs,
        context,
        context_lens,
        causal=False,
        num_frame_per_block=1,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.adaLN_modulation(e)).chunk(6, dim=-1)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens,
            freqs, causal=causal, num_frame_per_block=num_frame_per_block)
        # with amp.autocast(dtype=torch.float32):
        x = x + y * e[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
            # with amp.autocast(dtype=torch.float32):
            x = x + y * e[5]
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x

class WanAttentionDoubleStreamBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1_p1 = WanLayerNorm(dim, eps)
        self.norm1_p2 = WanLayerNorm(dim, eps)

        self.self_attn = WanSelfAttentionDoubleStream(dim, num_heads, qk_norm,
                                          eps)

        self.norm3_p1 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.norm3_p2 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
            
        self.cross_attn_p1 = WanT2VCrossAttention(dim,
                                                num_heads,
                                                qk_norm,
                                                eps)
        self.cross_attn_p2 = WanT2VCrossAttention(dim,
                                                num_heads,
                                                qk_norm,
                                                eps)

        self.norm2_p1 = WanLayerNorm(dim, eps)
        self.norm2_p2 = WanLayerNorm(dim, eps)

        self.ffn_p1 = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))
        self.ffn_p2 = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        self.adaLN_modulation_p1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 6 * self.dim, bias=True)
        )
        self.adaLN_modulation_p2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(self.dim, 6 * self.dim, bias=True)
        )

    def forward(
        self,
        x,
        e,
        seq_lens,
        freqs,
        context,
        context_lens,
        causal=False,
        num_frame_per_block=1,
        attention_mask_type=None,
        task_params=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        x_p1, x_p2 = x.chunk(2, dim=1)
        e_p1, e_p2 = e.chunk(2, dim=1)
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e_p1 = (self.adaLN_modulation_p1(e_p1)).chunk(6, dim=-1)
        e_p2 = (self.adaLN_modulation_p2(e_p2)).chunk(6, dim=-1)
        # assert e[0].dtype == torch.float32

        attn_input_p1 = self.norm1_p1(x_p1) * (1 + e_p1[1]) + e_p1[0]
        attn_input_p2 = self.norm1_p2(x_p2) * (1 + e_p2[1]) + e_p2[0]

        # self-attention
        y_p1, y_p2 = self.self_attn(
            attn_input_p1, attn_input_p2, seq_lens,
            freqs, causal=causal, num_frame_per_block=num_frame_per_block, attention_mask_type=attention_mask_type, task_params=task_params)
        # with amp.autocast(dtype=torch.float32):
        x_p1 = x_p1 + y_p1 * e_p1[2]
        x_p2 = x_p2 + y_p2 * e_p2[2]

        # cross-attention & ffn function
        def cross_attn_ffn(x_p1, x_p2, context, context_lens, e_p1, e_p2):
            x_p1 = x_p1 + self.cross_attn_p1(self.norm3_p1(x_p1), context, context_lens)
            x_p2 = x_p2 + self.cross_attn_p2(self.norm3_p2(x_p2), context, context_lens)
            y_p1, y_p2 = self.ffn_p1(self.norm2_p1(x_p1) * (1 + e_p1[4]) + e_p1[3]), self.ffn_p2(self.norm2_p2(x_p2) * (1 + e_p2[4]) + e_p2[3])
            # with amp.autocast(dtype=torch.float32):
            x_p1 = x_p1 + y_p1 * e_p1[5]
            x_p2 = x_p2 + y_p2 * e_p2[5]
            return x_p1, x_p2

        x_p1, x_p2 = cross_attn_ffn(x_p1, x_p2, context, context_lens, e_p1, e_p2)
        x = torch.cat((x_p1, x_p2), dim=1)
        return x