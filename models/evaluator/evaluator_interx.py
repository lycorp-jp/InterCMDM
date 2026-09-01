import time
import copy
import numpy as np
import os
from os.path import join as pjoin

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data.interx import Text2MotionDatasetV2HHI, collate_fn
from data.interx_utils import InterxNormalizerTorch, InterxNormalizerTorchSMPL
from utils.word_vectorizer import WordVectorizer, POS_enumerator
from einops import rearrange
from tqdm import tqdm

class EvaluationDataset(Dataset):

    def __init__(self, model, trans, w_vectorizer, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
        
        assert mm_num_samples < len(dataset)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=1, shuffle=True)
        self.w_vectorizer = w_vectorizer
        self.normalizer = InterxNormalizerTorchSMPL()
        self.max_motion_length = dataset.max_motion_length
        self.model = model.to(device)
        self.model.eval()
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()


        generated_motion = []
        mm_generated_motions = []
        
        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 6

        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # name, text, motion1, motion2, motion_lens = data
                word_emb, pos_ohot, caption, cap_lens, motions, motion_lens, tokens = data

                # motions = motions.reshape(motions.shape[0], motions.shape[1], motions.shape[2]//12, 12)
                motion1, motion2 = motions.split(motions.shape[-1]//2, dim=-1)
                
                word_emb = word_emb.detach().to(device).float()
                pos_ohot = pos_ohot.detach().to(device).float()
                
                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = []
                
                for t in range(repeat_times):
                    if trans is None:
                        motion_lens = motion_lens // 4 * 4
                        motion1_output, _, _ = self.model(motion1.float().to(device))
                        motion2_output, _, _ = self.model(motion2.float().to(device))
                    else:
                        ids_length = (motion_lens.detach().long().to(device)//4)
                        motion_lens = motion_lens // 4 * 4

                        # generate motion tokens
                        if opt.gen_react:
                            code_idx1, _ = self.model.encode(motion1.float().to(device))
                            motion_ids = trans.generate_reaction(caption, code_idx1[..., 0], ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        else:
                            motion_ids = trans.generate(caption, ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        
                        motion_ids1, motion_ids2 = motion_ids[:, :motion_ids.shape[1]//2], motion_ids[:, motion_ids.shape[1]//2:]
                        
                        # decode motion tokens
                        motion1_output = self.model.forward_decoder(motion_ids1.unsqueeze_(-1).to(device))
                        motion2_output = self.model.forward_decoder(motion_ids2.unsqueeze_(-1).to(device))
                    
                    if dataset.motion_rep == 'global':
                        motion1_output = self.normalizer.backward(motion1_output)
                        motion2_output = self.normalizer.backward(motion2_output)
                    motion_output = torch.cat([motion1_output, motion2_output], dim=-1)
                    # motion_output = motion_output.reshape(motion_output.shape[0], motion_output.shape[1], -1)
                    gen_motion_len = motion_output.shape[1]
                    
                    if trans is None:
                        save_vis_n = 10
                    else:
                        save_vis_n = 50
                    if i  < save_vis_n and opt.save_vis and t==0:
                        caption_dir = pjoin('/'.join(opt.vis_dir.split('/')[:-1]), 'captions')
                        os.makedirs(caption_dir, exist_ok=True)
                        
                        if trans is None:
                            gen_file_name = f"{file[:-4]}_{i:02d}_gen.npy"
                            gt_file_name = f"{file[:-4]}_{i:02d}_gt.npy"  
                        else:
                            gen_file_name = f"{file[:-4]}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_gen.npy"
                            gt_file_name = f"{file[:-4]}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_gt.npy"  
                        np.save(pjoin(opt.vis_dir, gen_file_name), motion_output.cpu().detach().numpy())
                        with open(pjoin(caption_dir, gen_file_name.replace('.npy', '.txt')), 'w') as f:
                            f.write(caption[0])

                        motion1 = self.normalizer.backward(motion1)
                        motion2 = self.normalizer.backward(motion2)

                        motion_input = torch.cat([motion1, motion2], dim=-1)
                        # motion_input = motion_input.reshape(motion_input.shape[0], motion_input.shape[1], -1)
                        np.save(pjoin(opt.vis_dir, gt_file_name), motion_input.cpu().detach().numpy())
                        with open(pjoin(caption_dir, gt_file_name.replace('.npy', '.txt')), 'w') as f:
                            f.write(caption[0])


                    if t == 0:
                        for b in range(motion_output.shape[0]):
                            sub_dict = {'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                        'length': motion_lens[b].item(),
                                        'cap_len': cap_lens[b].item(),
                                        'caption': caption[b],
                                        'tokens': tokens[b].split('_')}
                            generated_motion.append(sub_dict)
                    if is_mm:
                        for b in range(motion_output.shape[0]):
                            mm_motions.append({
                                'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                'length': motion_lens[b].item()
                            })
                
                if is_mm:
                    for b in range(len(mm_motions)):
                        mm_generated_motions.append({'caption': caption[b],
                                                    'tokens': tokens[b].split('_'),
                                                    'cap_len': cap_lens[b].item(),
                                                    'mm_motions': mm_motions[b]})

        self.generated_motion = generated_motion
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        # tokens = text_data['tokens']
        # if len(tokens) < self.opt.max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:self.opt.max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # print(tokens)
        # print(caption)
        # print(m_length)
        # print(self.opt.max_motion_length)
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class EvaluationDataset_cmdm(Dataset):

    def __init__(self, model, trans, w_vectorizer, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):

        assert mm_num_samples < len(dataset)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=1, shuffle=True)
        self.w_vectorizer = w_vectorizer
        self.normalizer = InterxNormalizerTorchSMPL()
        self.max_motion_length = dataset.max_motion_length
        self.model = model.to(device)
        self.model.eval()
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()


        generated_motion = []
        mm_generated_motions = []  # List to store multimodal samples (one dict per prompt)

        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 6

        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                # name, text, motion1, motion2, motion_lens = data
                word_emb, pos_ohot, caption, cap_lens, motions, motion_lens, tokens = data

                # motions = motions.reshape(motions.shape[0], motions.shape[1], motions.shape[2]//12, 12)
                motion1, motion2 = motions.split(motions.shape[-1]//2, dim=-1)

                j, d = motion1.shape[-2:]
                motion1 = rearrange(motion1, 'b l j d -> b l (j d)')
                motion2 = rearrange(motion2, 'b l j d -> b l (j d)')
                
                word_emb = word_emb.detach().to(device).float()
                pos_ohot = pos_ohot.detach().to(device).float()
                
                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions  = [[] for _ in range(len(motion_lens))]
                
                for t in range(repeat_times):
                    if trans is None:
                        motion1 = rearrange(motion1, 'b l d -> b l 1 d')
                        motion2 = rearrange(motion2, 'b l d -> b l 1 d')

                        motion_lens = motion_lens // 4 * 4

                        motion1_output= self.model(motion1.float().to(device))
                        motion2_output = self.model(motion2.float().to(device))

                        motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')
                        motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')

                    else:
                        ids_length = (motion_lens.detach().long().to(device)//4)

                        # generate motion tokens
                        if opt.gen_react:
                            raise NotImplementedError("Reaction generation is not supported for CMDM")
                        else:
                            motion_1_output, motion_2_output = trans.generate(caption, ids_length, cond_scale)

                        motion_1_output = rearrange(motion_1_output, f'b l 1 d -> {self.model.encode_dim}')
                        motion_2_output = rearrange(motion_2_output, f'b l 1 d -> {self.model.encode_dim}')
                        with torch.no_grad():
                            motion1_output = self.model.decode(motion_1_output)
                            motion2_output = self.model.decode(motion_2_output)
                        
                        motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')
                        motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')\

                    motion1_output = rearrange(motion1_output, 'b l (j d) -> b l j d', j=j, d=d)
                    motion2_output = rearrange(motion2_output, 'b l (j d) -> b l j d', j=j, d=d)
                    
                    motion1_output = self.normalizer.backward(motion1_output)
                    motion2_output = self.normalizer.backward(motion2_output)
                    
                    motion_output = torch.cat([motion1_output, motion2_output], dim=-1)
                    # motion_output = motion_output.reshape(motion_output.shape[0], motion_output.shape[1], -1)
                    gen_motion_len = motion_output.shape[1]
                    
                    if trans is None:
                        save_vis_n = 10
                    else:
                        save_vis_n = 50

                    if t == 0:
                        B = motion_output.shape[0]
                        for b in range(B):
                            sub_dict = {'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                        'length': motion_lens[b].item(),
                                        'cap_len': cap_lens[b].item(),
                                        'caption': caption[b],
                                        'tokens': tokens[b].split('_')}
                            generated_motion.append(sub_dict)

                    if is_mm:
                        for b in range(B):
                            mm_motions[b].append({
                                'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                'length': motion_lens[b].item()
                            })

                if is_mm:
                    for b in range(B):
                        mm_generated_motions.append({'caption': caption[b],
                                                     'tokens': tokens[b].split('_'),
                                                     'cap_len': cap_lens[b].item(),
                                                     'mm_motions': mm_motions[b]})

        self.generated_motion = generated_motion
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        # tokens = text_data['tokens']
        # if len(tokens) < self.opt.max_text_len:
        #     # pad with "unk"
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)
        # else:
        #     # crop
        #     tokens = tokens[:self.opt.max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        # print(tokens)
        # print(caption)
        # print(m_length)
        # print(self.opt.max_motion_length)
        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


class MMGeneratedDataset(Dataset):
    def __init__(self,  motion_dataset, w_vectorizer):
        self.max_motion_length = motion_dataset.max_motion_length
        self.dataset = motion_dataset.mm_generated_motions
        self.w_vectorizer = w_vectorizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        m_lens = []
        motions = []
        for mm_motion in mm_motions:
            m_lens.append(mm_motion['length'])
            motion = mm_motion['motion']
            if len(motion) < self.max_motion_length:
                motion = np.concatenate([motion,
                                         np.zeros((self.max_motion_length - len(motion), motion.shape[1], motion.shape[2]))
                                         ], axis=0)
            motion = motion[None, :]
            motions.append(motion)
        m_lens = np.array(m_lens, dtype=int)
        motions = np.concatenate(motions, axis=0)
        sort_indx = np.argsort(m_lens)[::-1].copy()
        # print(m_lens)
        # print(sort_indx)
        # print(m_lens[sort_indx])
        m_lens = m_lens[sort_indx]
        motions = motions[sort_indx]
        return motions, m_lens


def get_motion_loader(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    # Currently the configurations of two datasets are almost the same
    start = time.time()
    w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab')
    dataset = EvaluationDataset(model, trans, 
                                w_vectorizer, ground_truth_dataset,
                                device, 
                                mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, 
                                file=file, opt=opt, 
                                time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print(f'Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


def get_motion_loader_cmdm(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    # Currently the configurations of two datasets are almost the same
    start = time.time()
    w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab')
    dataset = EvaluationDataset_cmdm(model, trans,
                                w_vectorizer, ground_truth_dataset,
                                device,
                                mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats,
                                file=file, opt=opt,
                                time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print(f'Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


class EvaluationDataset_cmdm_gt_p1(Dataset):
    """Evaluation dataset where Person 1's motion is fixed as ground truth
    and only Person 2's motion is generated (Inter-X)."""

    def __init__(self, model, trans, w_vectorizer, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
        assert mm_num_samples < len(dataset)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=1, shuffle=True)
        self.w_vectorizer = w_vectorizer
        self.normalizer = InterxNormalizerTorchSMPL()
        self.max_motion_length = dataset.max_motion_length
        self.model = model.to(device)
        self.model.eval()
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()

        generated_motion = []
        mm_generated_motions = []

        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, motion_lens, tokens = data
                motion1, motion2 = motions.split(motions.shape[-1] // 2, dim=-1)

                j, d = motion1.shape[-2:]
                motion1_flat = rearrange(motion1, 'b l j d -> b l (j d)')
                motion2_flat = rearrange(motion2, 'b l j d -> b l (j d)')

                word_emb = word_emb.detach().to(device).float()
                pos_ohot = pos_ohot.detach().to(device).float()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = [[] for _ in range(len(motion_lens))]

                for t in range(repeat_times):
                    if trans is None:
                        motion1_in = rearrange(motion1_flat, 'b l d -> b l 1 d')
                        motion2_in = rearrange(motion2_flat, 'b l d -> b l 1 d')
                        motion_lens = motion_lens // 4 * 4
                        motion1_output = self.model(motion1_in.float().to(device))
                        motion2_output = self.model(motion2_in.float().to(device))
                        motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')
                        motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')
                    else:
                        ids_length = (motion_lens.detach().long().to(device) // 4)

                        motion1_in = rearrange(motion1_flat.float().to(device), 'b l d -> b l 1 d')
                        gt_latent_1 = self.model.encode(motion1_in)
                        gt_latent_1 = rearrange(gt_latent_1, f'{self.model.encode_dim} -> b l 1 d', d=self.model.output_emb_width)

                        _, latent_2 = trans.generate_with_gt_person1(caption, ids_length, gt_latent_1, cond_scale)

                        latent_2_dec = rearrange(latent_2, f'b l 1 d -> {self.model.encode_dim}')
                        motion2_output = self.model.decode(latent_2_dec)
                        motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')

                        motion1_output = motion1_flat.float().to(device)

                    motion1_out_j = rearrange(motion1_output, 'b l (j d) -> b l j d', j=j, d=d)
                    motion2_out_j = rearrange(motion2_output, 'b l (j d) -> b l j d', j=j, d=d)

                    motion1_out_j = self.normalizer.backward(motion1_out_j)
                    motion2_out_j = self.normalizer.backward(motion2_out_j)

                    motion_output = torch.cat([motion1_out_j, motion2_out_j], dim=-1)
                    gen_motion_len = motion_output.shape[1]

                    if t == 0:
                        B = motion_output.shape[0]
                        for b in range(B):
                            sub_dict = {'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                        'length': motion_lens[b].item(),
                                        'cap_len': cap_lens[b].item(),
                                        'caption': caption[b],
                                        'tokens': tokens[b].split('_')}
                            generated_motion.append(sub_dict)

                    if is_mm:
                        for b in range(B):
                            mm_motions[b].append({
                                'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                'length': motion_lens[b].item()
                            })

                if is_mm:
                    for b in range(B):
                        mm_generated_motions.append({'caption': caption[b],
                                                     'tokens': tokens[b].split('_'),
                                                     'cap_len': cap_lens[b].item(),
                                                     'mm_motions': mm_motions[b]})

        self.generated_motion = generated_motion
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


def get_motion_loader_cmdm_gt_p1(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    start = time.time()
    w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab')
    dataset = EvaluationDataset_cmdm_gt_p1(model, trans,
                                           w_vectorizer, ground_truth_dataset,
                                           device,
                                           mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats,
                                           file=file, opt=opt,
                                           time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print(f'GT-P1 Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


class EvaluationDataset_cmdm_gt_p2(Dataset):
    """Evaluation dataset where Person 2's motion is fixed as ground truth
    and only Person 1's motion is generated (Inter-X)."""

    def __init__(self, model, trans, w_vectorizer, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
        assert mm_num_samples < len(dataset)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=1, shuffle=True)
        self.w_vectorizer = w_vectorizer
        self.normalizer = InterxNormalizerTorchSMPL()
        self.max_motion_length = dataset.max_motion_length
        self.model = model.to(device)
        self.model.eval()
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()

        generated_motion = []
        mm_generated_motions = []

        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, motion_lens, tokens = data
                motion1, motion2 = motions.split(motions.shape[-1] // 2, dim=-1)

                j, d = motion1.shape[-2:]
                motion1_flat = rearrange(motion1, 'b l j d -> b l (j d)')
                motion2_flat = rearrange(motion2, 'b l j d -> b l (j d)')

                word_emb = word_emb.detach().to(device).float()
                pos_ohot = pos_ohot.detach().to(device).float()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions = [[] for _ in range(len(motion_lens))]

                for t in range(repeat_times):
                    if trans is None:
                        motion1_in = rearrange(motion1_flat, 'b l d -> b l 1 d')
                        motion2_in = rearrange(motion2_flat, 'b l d -> b l 1 d')
                        motion_lens = motion_lens // 4 * 4
                        motion1_output = self.model(motion1_in.float().to(device))
                        motion2_output = self.model(motion2_in.float().to(device))
                        motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')
                        motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')
                    else:
                        ids_length = (motion_lens.detach().long().to(device) // 4)

                        motion2_in = rearrange(motion2_flat.float().to(device), 'b l d -> b l 1 d')
                        gt_latent_2 = self.model.encode(motion2_in)
                        gt_latent_2 = rearrange(gt_latent_2, f'{self.model.encode_dim} -> b l 1 d', d=self.model.output_emb_width)

                        latent_1, _ = trans.generate_with_gt_person2(caption, ids_length, gt_latent_2, cond_scale)

                        latent_1_dec = rearrange(latent_1, f'b l 1 d -> {self.model.encode_dim}')
                        motion1_output = self.model.decode(latent_1_dec)
                        motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')

                        motion2_output = motion2_flat.float().to(device)

                    motion1_out_j = rearrange(motion1_output, 'b l (j d) -> b l j d', j=j, d=d)
                    motion2_out_j = rearrange(motion2_output, 'b l (j d) -> b l j d', j=j, d=d)

                    motion1_out_j = self.normalizer.backward(motion1_out_j)
                    motion2_out_j = self.normalizer.backward(motion2_out_j)

                    motion_output = torch.cat([motion1_out_j, motion2_out_j], dim=-1)

                    if t == 0:
                        B = motion_output.shape[0]
                        for b in range(B):
                            sub_dict = {'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                        'length': motion_lens[b].item(),
                                        'cap_len': cap_lens[b].item(),
                                        'caption': caption[b],
                                        'tokens': tokens[b].split('_')}
                            generated_motion.append(sub_dict)

                    if is_mm:
                        for b in range(B):
                            mm_motions[b].append({
                                'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                'length': motion_lens[b].item()
                            })

                if is_mm:
                    for b in range(B):
                        mm_generated_motions.append({'caption': caption[b],
                                                     'tokens': tokens[b].split('_'),
                                                     'cap_len': cap_lens[b].item(),
                                                     'mm_motions': mm_motions[b]})

        self.generated_motion = generated_motion
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motion)

    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


def get_motion_loader_cmdm_gt_p2(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    start = time.time()
    w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab')
    dataset = EvaluationDataset_cmdm_gt_p2(model, trans,
                                           w_vectorizer, ground_truth_dataset,
                                           device,
                                           mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats,
                                           file=file, opt=opt,
                                           time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print(f'GT-P2 Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


class EvaluationDataset_cmdm_raw(Dataset):
    """Evaluation dataset for DiT models that work directly on raw 336D motion features (Inter-X)"""

    def __init__(self, trans, w_vectorizer, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):

        assert mm_num_samples < len(dataset)
        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=1, shuffle=True)
        self.w_vectorizer = w_vectorizer
        self.normalizer = InterxNormalizerTorch()
        self.max_motion_length = dataset.max_motion_length
        self.trans = trans.to(device)
        self.trans.eval()


        generated_motion = []
        mm_generated_motions = []

        mm_idxs = np.random.choice(len(dataset), mm_num_samples, replace=False)
        mm_idxs = np.sort(mm_idxs)
        min_mov_length = 6

        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                word_emb, pos_ohot, caption, cap_lens, motions, motion_lens, tokens = data

                motion1, motion2 = motions.split(motions.shape[-1]//2, dim=-1)

                j, d = motion1.shape[-2:]
                motion1 = rearrange(motion1, 'b l j d -> b l (j d)')
                motion2 = rearrange(motion2, 'b l j d -> b l (j d)')

                word_emb = word_emb.detach().to(device).float()
                pos_ohot = pos_ohot.detach().to(device).float()

                mm_num_now = len(mm_generated_motions)
                is_mm = True if ((mm_num_now < mm_num_samples) and (i == mm_idxs[mm_num_now])) else False
                repeat_times = mm_num_repeats if is_mm else 1
                mm_motions  = [[] for _ in range(len(motion_lens))]

                for t in range(repeat_times):
                    ids_length = motion_lens.detach().long().to(device)

                    # Generate motion directly in raw 336D space
                    if opt.gen_react:
                        raise NotImplementedError("Reaction generation is not supported for raw CMDM")
                    else:
                        motion_1_output, motion_2_output = trans.generate(caption, ids_length, cond_scale)

                    # Motion is already in raw 336D space, no need to decode
                    motion1_output = rearrange(motion_1_output, 'b l 1 d -> b l d')
                    motion2_output = rearrange(motion_2_output, 'b l 1 d -> b l d')

                    motion1_output = rearrange(motion1_output, 'b l (j d) -> b l j d', j=j, d=d)
                    motion2_output = rearrange(motion2_output, 'b l (j d) -> b l j d', j=j, d=d)

                    if dataset.motion_rep == 'global':
                        motion1_output = self.normalizer.backward(motion1_output)
                        motion2_output = self.normalizer.backward(motion2_output)
                    motion_output = torch.cat([motion1_output, motion2_output], dim=-1)
                    gen_motion_len = motion_output.shape[1]

                    save_vis_n = 50

                    if t == 0:
                        B = motion_output.shape[0]
                        for b in range(B):
                            sub_dict = {'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                        'length': motion_lens[b].item(),
                                        'cap_len': cap_lens[b].item(),
                                        'caption': caption[b],
                                        'tokens': tokens[b].split('_')}
                            generated_motion.append(sub_dict)

                    if is_mm:
                        for b in range(B):
                            mm_motions[b].append({
                                'motion': motion_output[b, :motion_lens[b].item()].detach().cpu().numpy(),
                                'length': motion_lens[b].item()
                            })

                if is_mm:
                    for b in range(B):
                        mm_generated_motions.append({'caption': caption[b],
                                                     'tokens': tokens[b].split('_'),
                                                     'cap_len': cap_lens[b].item(),
                                                     'mm_motions': mm_motions[b]})

        self.generated_motion = generated_motion
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motion)


    def __getitem__(self, item):
        data = self.generated_motion[item]
        motion, m_length, caption, tokens = data['motion'], data['length'], data['caption'], data['tokens']
        sent_len = data['cap_len']
        pos_one_hots = []
        word_embeddings = []
        for token in tokens:
            try:
                word_emb, pos_oh = self.w_vectorizer[token]
            except:
                word_emb, pos_oh = self.w_vectorizer['unk/OTHER']
            pos_one_hots.append(pos_oh[None, :])
            word_embeddings.append(word_emb[None, :])
        pos_one_hots = np.concatenate(pos_one_hots, axis=0)
        word_embeddings = np.concatenate(word_embeddings, axis=0)

        if m_length < self.max_motion_length:
            motion = np.concatenate([motion,
                                     np.zeros((self.max_motion_length - m_length, motion.shape[1], motion.shape[2]))
                                     ], axis=0)
        return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, '_'.join(tokens)


def get_motion_loader_cmdm_raw(batch_size, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    """Motion loader for DiT models that work directly on raw 336D motion features (Inter-X)"""
    start = time.time()
    w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab')
    dataset = EvaluationDataset_cmdm_raw(trans,
                                w_vectorizer, ground_truth_dataset,
                                device,
                                mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats,
                                file=file, opt=opt,
                                time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset, w_vectorizer)

    motion_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, drop_last=True, num_workers=4)
    mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)

    print(f'Raw Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


def get_dataset_motion_loader(opt, batch_size, normalize=False):
    opt = copy.deepcopy(opt)

    if opt.dataset_name == 'interx':
        print('Loading dataset %s ...' % opt.dataset_name)


        dataset = Text2MotionDatasetV2HHI(opt, 
                                            pjoin(opt.data_root, 'splits/test.txt'), 
                                            WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab'), 
                                            pjoin(opt.motion_dir, 'test.h5'), normalize=normalize)
        dataloader = DataLoader(dataset, batch_size=batch_size, 
                                 num_workers=4, drop_last=True, collate_fn=collate_fn, shuffle=True)

    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


def init_weight(m):
    if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose1d):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class MovementConvEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MovementConvEncoder, self).__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return self.out_net(outputs)

class TextEncoderBiGRUCo(nn.Module):
    def __init__(self, word_size, pos_size, hidden_size, output_size, device):
        super(TextEncoderBiGRUCo, self).__init__()
        self.device = device

        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.pos_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))
    
    def forward(self, word_embs, pos_onehot, cap_lens):
        num_samples = word_embs.shape[0]

        pos_embs = self.pos_emb(pos_onehot)
        inputs = word_embs + pos_embs
        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = cap_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)

class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(MotionEncoderBiGRUCo, self).__init__()
        self.device = device

        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size*2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size)
        )

        self.input_emb.apply(init_weight)
        self.output_net.apply(init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn((2, 1, self.hidden_size), requires_grad=True))

    # input(batch_size, seq_len, dim)
    def forward(self, inputs, m_lens):
        num_samples = inputs.shape[0]

        input_embs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, num_samples, 1)

        cap_lens = m_lens.data.tolist()
        emb = pack_padded_sequence(input_embs, cap_lens, batch_first=True)

        gru_seq, gru_last = self.gru(emb, hidden)

        gru_last = torch.cat([gru_last[0], gru_last[1]], dim=-1)

        return self.output_net(gru_last)
    
def build_models(opt):
    movement_enc = MovementConvEncoder(opt.dim_pose, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
                                  pos_size=opt.dim_pos_ohot,
                                  hidden_size=opt.dim_text_hidden,
                                  output_size=opt.dim_coemb_hidden,
                                  device=opt.device)

    motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
                                      hidden_size=opt.dim_motion_hidden,
                                      output_size=opt.dim_coemb_hidden,
                                      device=opt.device)

    checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
                            map_location=opt.device)
    movement_enc.load_state_dict(checkpoint['movement_encoder'])
    text_enc.load_state_dict(checkpoint['text_encoder'])
    motion_enc.load_state_dict(checkpoint['motion_encoder'])
    print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return text_enc, motion_enc, movement_enc


class EvaluatorModelWrapper(object):

    def __init__(self, opt):

        opt.dim_pose = 56 * 12

        opt.dim_word = 300
        opt.max_motion_length = 196
        opt.dim_pos_ohot = len(POS_enumerator)
        opt.dim_motion_hidden = 1024
        opt.max_text_len = 20
        opt.dim_text_hidden = 512
        opt.dim_coemb_hidden = 512
        if opt.dataset_name == 'hhi':
            opt.max_motion_length = 150
            opt.max_text_len = 35

        self.text_encoder, self.motion_encoder, self.movement_encoder = build_models(opt)
        self.opt = opt
        self.device = opt.device

        self.text_encoder.to(opt.device)
        self.motion_encoder.to(opt.device)
        self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def prepare_motion(self, motions):
        # motion1, motion2 = motions.split(motions.shape[-1]//2, dim=-1)

        # rot1 = motion1[:,:,:,6:]
        # trans1 = motion1[:,:,[0],:6]
        # motion1 = torch.cat([rot1, trans1], dim=-2)

        # rot2 = motion2[:,:,:,6:]
        # trans2 = motion2[:,:,[0],:6]
        # motion2 = torch.cat([rot2, trans2], dim=-2)

        # motions = torch.cat([motion1, motion2], dim=-1)

        motions[:,:,-1,9:] = 0
        motions[:,:,-1,3:6] = 0
        motions=motions.reshape(motions.shape[0], motions.shape[1],-1)
        return motions
    
    def get_co_embeddings(self, batch):
        word_embs, pos_ohot, _, cap_lens, motions, m_lens, _ = batch
        
        with torch.no_grad():
            word_embs = word_embs.detach().to(self.device).float()
            pos_ohot = pos_ohot.detach().to(self.device).float()
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            motions = self.prepare_motion(motions)
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)

            '''Text Encoding'''
            text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
            text_embedding = text_embedding[align_idx]
        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch):
        try:
            _, _, _, sent_lens, motions, m_lens, _ = batch
        except ValueError:
            motions, m_lens = batch
        if len(motions.shape) == 5:
            motions = motions[0]
            m_lens = m_lens[0]
        with torch.no_grad():
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            m_lens = m_lens[align_idx]

            '''Movement Encoding'''
            motions = self.prepare_motion(motions)
            movements = self.movement_encoder(motions).detach()
            m_lens = m_lens // self.opt.unit_length
            motion_embedding = self.motion_encoder(movements, m_lens)
        return motion_embedding
