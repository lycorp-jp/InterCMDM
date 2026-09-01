from os.path import join as pjoin
import torch
from torch.utils.data import Dataset, DataLoader
from data.interhuman import InterHumanDataset
from data.utils import MotionNormalizer
from utils.plot_script import preprocess_plot_motion
# from models import *
import copy
import random
import time
import numpy as np
from models.evaluator.evaluator_models import InterCLIP
from tqdm import tqdm
from einops import rearrange

class EvaluationDataset(Dataset):

    def __init__(self, model, trans, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()

        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        random.shuffle(idxs)
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in enumerate(dataloader):
                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    num_repeats = mm_num_repeats
                else:
                    num_repeats = 1
                    # text = list(text) * mm_num_repeats
                    # motion_lens = torch.tensor([motion_lens[0].item()] * mm_num_repeats)
                
                if trans is None:
                    motion1_output, _, _ = self.model(motion1.float().to(device))
                    motion2_output, _, _ = self.model(motion2.float().to(device))
                
                else:
                    ids_length = (motion_lens.detach().long().to(device)//4)

                    for num_repeat in range(num_repeats):
                        if opt.gen_react:
                            code_idx1, _ = self.model.encode(motion1.float().to(device))
                            motion_ids = trans.generate_reaction(text, code_idx1[..., 0], ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        else:
                            motion_ids = trans.generate(text, ids_length, time_steps, cond_scale, topk_filter_thres=topkr, temperature=1)
                        
                        motion_ids1, motion_ids2 = motion_ids[:, :motion_ids.shape[1]//2], motion_ids[:, motion_ids.shape[1]//2:]
                       
                        motion1_output_one = self.model.forward_decoder(motion_ids1.unsqueeze_(-1).to(device))
                        motion2_output_one = self.model.forward_decoder(motion_ids2.unsqueeze_(-1).to(device))
                        
                        if num_repeat == 0:
                            motion1_output = motion1_output_one
                            motion2_output = motion2_output_one
                        else:
                            motion1_output = torch.cat((motion1_output, motion1_output_one), dim=0)
                            motion2_output = torch.cat((motion2_output, motion2_output_one), dim=0)

                padding_len = motion1.shape[1] - motion1_output.shape[1]
                B, D = motion1_output.shape[0], motion1_output.shape[2]
                padding_zeros = torch.zeros((B, padding_len, D)).to(device)
                motion1_output = torch.concat((motion1_output, padding_zeros), dim=1)
                motion2_output = torch.concat((motion2_output, padding_zeros), dim=1)
                
                if opt.gen_react:
                    batch.update({"output": torch.cat([motion1.to(device), motion2_output], dim=-1)})
                else:
                    batch.update({"output": torch.cat([motion1_output, motion2_output], dim=-1)})
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())
                # motions_output = motions_output.cpu().detach().numpy()

                if trans is None:
                    save_vis_n = 10
                else:
                    save_vis_n = 20

                if i  < save_vis_n and opt.save_vis:
                    motions_input = torch.cat([motion1, motion2], dim=-1)[0]
                    motions_input = motions_input.reshape(motions_input.shape[0], 2, -1)
                    motions_input = self.normalizer.backward(motions_input.cpu().detach().numpy())
                    # motions_input = motions_input.cpu().detach().numpy()
                    
                    preprocess_plot_motion(motions_input[:motion_lens[0].item(), :, :],  text[0],
                                           opt.vis_dir, opt.npy_dir,
                                           f"{file.split('.')[0]}_{i:02d}_gt", foot_ik=False)
                    if trans is None:
                        gen_file_name = f"{file.split('.')[0]}_{i:02d}_gen"
                    else:
                        gen_file_name = f"{file.split('.')[0]}_ts{time_steps}_cs{cond_scale}_topkr{topkr}_{i:02d}_gen"
                        
                    preprocess_plot_motion(motions_output[0][:motion_lens[0].item(), :, :], text[0],
                                           opt.vis_dir, opt.npy_dir,
                                           gen_file_name, foot_ik=True,)
                
                # if i >= save_vis_n and opt.save_vis:
                #     exit()

                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length

                # Get original batch size (before replication for multimodality)
                original_B = len(text)
                for b in range(original_B):
                    if i in mm_idxs:
                        # For multimodal batches, extract all replications for this prompt
                        mm_sub_dict = {'mm_motions': motions_output[b*num_repeats:(b+1)*num_repeats, :, :],
                                       'motion_lens': motion_lens[b].item(),
                                       'text': text[b]}
                        mm_generated_motions.append(mm_sub_dict)

                        # For generated_motions, take only the first replication
                        sub_dict = {'motion1': motions_output[b*num_repeats, :, 0],
                                    'motion2': motions_output[b*num_repeats, :, 1],
                                    'motion_lens': motion_lens[b].item(),
                                    'text': text[b]}
                    else:
                        # No replications, use directly
                        sub_dict = {'motion1': motions_output[b, :, 0],
                                    'motion2': motions_output[b, :, 1],
                                    'motion_lens': motion_lens[b].item(),
                                    'text': text[b]}
                    generated_motions.append(sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


class EvaluationDataset_cmdm(Dataset):

    def __init__(self, model, trans, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr, sampler='euler'):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        self.sampler = sampler
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()

        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        # Pre-process all target captions
        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    num_repeats = mm_num_repeats
                else:
                    num_repeats = 1
                
                if trans is None:
                    motion1 = rearrange(motion1, 'b l d -> b l 1 d').float().to(device)
                    motion2 = rearrange(motion2, 'b l d -> b l 1 d').float().to(device)

                    motion_lens = motion_lens // 4 * 4

                    motion1_output = self.model(motion1)
                    motion2_output = self.model(motion2)

                    motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')
                    motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')
                
                else:
                    ids_length = (motion_lens.detach().long().to(device)//4)

                    for num_repeat in range(num_repeats):
                        if opt.gen_react:
                            raise NotImplementedError("Reaction generation is not supported for CMDM")
                        else:
                            motion_1_output, motion_2_output = trans.generate(text, ids_length, cond_scale)

                        motion_1_output = rearrange(motion_1_output, f'b l 1 d -> {self.model.encode_dim}')
                        motion_2_output = rearrange(motion_2_output, f'b l 1 d -> {self.model.encode_dim}')
                        with torch.no_grad():
                            motion_1_output = self.model.decode(motion_1_output)
                            motion_2_output = self.model.decode(motion_2_output)
                        
                        motion_1_output = rearrange(motion_1_output, 'b l 1 d -> b l d')
                        motion_2_output = rearrange(motion_2_output, 'b l 1 d -> b l d')
                    
                        if num_repeat == 0:
                            motion1_output = motion_1_output
                            motion2_output = motion_2_output
                        else:
                            motion1_output = torch.cat((motion1_output, motion_1_output), dim=0)
                            motion2_output = torch.cat((motion2_output, motion_2_output), dim=0)

                padding_len = motion1.shape[1] - motion1_output.shape[1]
                B, D = motion1_output.shape[0], motion1_output.shape[2]
                padding_zeros = torch.zeros((B, padding_len, D)).to(device)
                motion1_output = torch.concat((motion1_output, padding_zeros), dim=1)
                motion2_output = torch.concat((motion2_output, padding_zeros), dim=1)
                
                if opt.gen_react:
                    raise NotImplementedError("Reaction generation is not supported for CMDM")
                else:
                    batch.update({"output": torch.cat([motion1_output, motion2_output], dim=-1)})
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                if opt.save_vis and i == 0:

                    if trans is None:
                        save_vis_n = 10
                    else:
                        save_vis_n = 20

                    for b in range(B):
                        if b >= save_vis_n:
                            break

                        motions_input = torch.cat([motion1, motion2], dim=-1)
                        motions_input = motions_input.reshape(motions_input.shape[0], motions_input.shape[1], 2, -1)
                        motions_input = self.normalizer.backward(motions_input.cpu().detach().numpy())

                        preprocess_plot_motion(motions_input[b, :motion_lens[b].item()],  text[b],
                                            opt.vis_dir, opt.npy_dir,
                                            f"{file.split('.')[0]}_{b:02d}_gt", foot_ik=False)

                        gen_file_name = f"{file.split('.')[0]}_{b:02d}_gen"
                            
                        preprocess_plot_motion(motions_output[b, :motion_lens[b].item()], text[b],
                                            opt.vis_dir, opt.npy_dir,
                                            gen_file_name, foot_ik=True,)

                    raise Exception("Visualization saved")


                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length

                # Get original batch size (before replication for multimodality)
                original_B = len(text)
                for b in range(original_B):
                    m_len = motion_lens[b]
                    if i in mm_idxs:
                        # For multimodal batches, extract all replications for this prompt
                        mm_sub_dict = {'mm_motions': motions_output[b*num_repeats:(b+1)*num_repeats, :, :],
                                       'motion_lens': m_len,
                                       'text': text[b]}
                        print (f"mm_sub_dict: {mm_sub_dict['mm_motions'].shape}")
                        mm_generated_motions.append(mm_sub_dict)

                        # For generated_motions, take only the first replication
                        sub_dict = {'motion1': motions_output[b*num_repeats, :, 0],
                                    'motion2': motions_output[b*num_repeats, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    else:
                        # No replications, use directly
                        sub_dict = {'motion1': motions_output[b, :, 0],
                                    'motion2': motions_output[b, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    generated_motions.append(sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


class MMGeneratedDataset(Dataset):
    def __init__(self, motion_dataset):
        self.dataset = motion_dataset.mm_generated_motions

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        data = self.dataset[item]
        mm_motions = data['mm_motions']
        motion_lens = data['motion_lens']
        mm_motions1 = mm_motions[:,:,0]
        mm_motions2 = mm_motions[:,:,1]
        text = data['text']
        motion_lens = np.array([motion_lens]*mm_motions1.shape[0])
        return "mm_generated", text, mm_motions1, mm_motions2, motion_lens


def get_dataset_motion_loader(opt, batch_size, normalize=False):
    opt = copy.deepcopy(opt)
    # Configurations of T2M dataset and KIT dataset is almost the same
    if opt.dataset_name == 'interhuman' or opt.dataset_name == 'joint':
        print('Loading dataset %s ...' % opt.dataset_name)

        dataset = InterHumanDataset(opt, normalize=normalize)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')

    print('Ground Truth Dataset Loading Completed!!!')
    return dataloader, dataset


def get_dataset_motion_loader_GT(opt, batch_size, normalize=False):
    opt = copy.deepcopy(opt)
    if opt.dataset_name == 'interhuman' or opt.dataset_name == 'joint':
        print('Loading dataset %s ...' % opt.dataset_name)
        dataset = InterHumanDatasetJoint(opt, normalize=normalize)
        dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=0, drop_last=True, shuffle=True)
    else:
        raise KeyError('Dataset not Recognized !!')
    return dataloader, dataset

def get_motion_loader(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    # Currently the configurations of two datasets are almost the same
    start = time.time()
    dataset = EvaluationDataset(model, trans, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, file=file, opt=opt, time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    print(f'Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


def get_motion_loader_cmdm(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr, sampler='euler'):
    # Currently the configurations of two datasets are almost the same
    start = time.time()
    dataset = EvaluationDataset_cmdm(model, trans, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, file=file, opt=opt, time_steps=time_steps, cond_scale=cond_scale, topkr=topkr, sampler=sampler)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    print(f'Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


class EvaluationDataset_cmdm_gt_p1(Dataset):
    """Evaluation dataset where Person 1's motion is fixed as ground truth
    and only Person 2's motion is generated."""

    def __init__(self, model, trans, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr, sampler='euler'):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        self.sampler = sampler
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()

        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    num_repeats = mm_num_repeats
                else:
                    num_repeats = 1

                if trans is None:
                    motion1_input = rearrange(motion1, 'b l d -> b l 1 d').float().to(device)
                    motion2_input = rearrange(motion2, 'b l d -> b l 1 d').float().to(device)
                    motion1_output = self.model(motion1_input)
                    motion2_output = self.model(motion2_input)
                    motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')
                    motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')
                else:
                    ids_length = (motion_lens.detach().long().to(device) // 4)

                    motion1_input = rearrange(motion1.float().to(device), 'b l d -> b l 1 d')
                    gt_latent_1 = self.model.encode(motion1_input)
                    gt_latent_1 = rearrange(gt_latent_1, f'{self.model.encode_dim} -> b l 1 d', d=self.model.output_emb_width)

                    for num_repeat in range(num_repeats):
                        _, latent_2 = trans.generate_with_gt_person1(text, ids_length, gt_latent_1, cond_scale)

                        latent_2_dec = rearrange(latent_2, f'b l 1 d -> {self.model.encode_dim}')
                        with torch.no_grad():
                            motion_2_out = self.model.decode(latent_2_dec)
                        motion_2_out = rearrange(motion_2_out, 'b l 1 d -> b l d')

                        if num_repeat == 0:
                            motion2_output = motion_2_out
                        else:
                            motion2_output = torch.cat((motion2_output, motion_2_out), dim=0)

                    motion1_output = motion1.float().to(device)
                    if num_repeats > 1:
                        motion1_output = motion1_output.repeat(num_repeats, 1, 1)

                padding_len = motion1.shape[1] - motion2_output.shape[1]
                if padding_len > 0:
                    B, D = motion2_output.shape[0], motion2_output.shape[2]
                    padding_zeros = torch.zeros((B, padding_len, D)).to(device)
                    motion2_output = torch.concat((motion2_output, padding_zeros), dim=1)
                if motion1_output.shape[1] != motion1.shape[1]:
                    padding_len_1 = motion1.shape[1] - motion1_output.shape[1]
                    if padding_len_1 > 0:
                        B1, D1 = motion1_output.shape[0], motion1_output.shape[2]
                        motion1_output = torch.concat((motion1_output, torch.zeros((B1, padding_len_1, D1)).to(device)), dim=1)

                batch.update({"output": torch.cat([motion1_output, motion2_output], dim=-1)})
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                if opt.save_vis and i == 0:
                    save_vis_n = 20 if trans is not None else 10
                    for b in range(min(motions_output.shape[0], save_vis_n)):
                        motions_input = torch.cat([motion1, motion2], dim=-1)
                        motions_input = motions_input.reshape(motions_input.shape[0], motions_input.shape[1], 2, -1)
                        motions_input = self.normalizer.backward(motions_input.cpu().detach().numpy())

                        preprocess_plot_motion(motions_input[b, :motion_lens[b].item()], text[b],
                                               opt.vis_dir, opt.npy_dir,
                                               f"{file.split('.')[0]}_{b:02d}_gt", foot_ik=False)
                        gen_file_name = f"{file.split('.')[0]}_gtp1_{b:02d}_gen"
                        preprocess_plot_motion(motions_output[b, :motion_lens[b].item()], text[b],
                                               opt.vis_dir, opt.npy_dir,
                                               gen_file_name, foot_ik=True)

                B, T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length

                # Get original batch size (before replication for multimodality)
                original_B = len(text)
                for b in range(original_B):
                    m_len = motion_lens[b]
                    if i in mm_idxs:
                        # For multimodal batches, extract all replications for this prompt
                        mm_sub_dict = {'mm_motions': motions_output[b * num_repeats:(b + 1) * num_repeats, :, :],
                                       'motion_lens': m_len,
                                       'text': text[b]}
                        mm_generated_motions.append(mm_sub_dict)

                        # For generated_motions, take only the first replication
                        sub_dict = {'motion1': motions_output[b * num_repeats, :, 0],
                                    'motion2': motions_output[b * num_repeats, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    else:
                        # No replications, use directly
                        sub_dict = {'motion1': motions_output[b, :, 0],
                                    'motion2': motions_output[b, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    generated_motions.append(sub_dict)

        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


def get_motion_loader_cmdm_gt_p1(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr, sampler='euler'):
    start = time.time()
    dataset = EvaluationDataset_cmdm_gt_p1(model, trans, ground_truth_dataset, device,
                                           mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats,
                                           file=file, opt=opt, time_steps=time_steps,
                                           cond_scale=cond_scale, topkr=topkr, sampler=sampler)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    print(f'GT-P1 Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


class EvaluationDataset_cmdm_gt_p2(Dataset):
    """Evaluation dataset where Person 2's motion is fixed as ground truth
    and only Person 1's motion is generated (reaction of Person 1 to Person 2)."""

    def __init__(self, model, trans, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr, sampler='euler'):
        self.normalizer = MotionNormalizer()
        self.model = model.to(device)
        self.model.eval()
        self.sampler = sampler
        if trans is not None:
            self.trans = trans.to(device)
            self.trans.eval()

        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []
        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):
                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    num_repeats = mm_num_repeats
                else:
                    num_repeats = 1

                if trans is None:
                    motion1_input = rearrange(motion1, 'b l d -> b l 1 d').float().to(device)
                    motion2_input = rearrange(motion2, 'b l d -> b l 1 d').float().to(device)
                    motion1_output = self.model(motion1_input)
                    motion2_output = self.model(motion2_input)
                    motion1_output = rearrange(motion1_output, 'b l 1 d -> b l d')
                    motion2_output = rearrange(motion2_output, 'b l 1 d -> b l d')
                else:
                    ids_length = (motion_lens.detach().long().to(device) // 4)

                    motion2_input = rearrange(motion2.float().to(device), 'b l d -> b l 1 d')
                    gt_latent_2 = self.model.encode(motion2_input)
                    gt_latent_2 = rearrange(gt_latent_2, f'{self.model.encode_dim} -> b l 1 d', d=self.model.output_emb_width)

                    for num_repeat in range(num_repeats):
                        latent_1, _ = trans.generate_with_gt_person2(text, ids_length, gt_latent_2, cond_scale)

                        latent_1_dec = rearrange(latent_1, f'b l 1 d -> {self.model.encode_dim}')
                        with torch.no_grad():
                            motion_1_out = self.model.decode(latent_1_dec)
                        motion_1_out = rearrange(motion_1_out, 'b l 1 d -> b l d')

                        if num_repeat == 0:
                            motion1_output = motion_1_out
                        else:
                            motion1_output = torch.cat((motion1_output, motion_1_out), dim=0)

                    motion2_output = motion2.float().to(device)
                    if num_repeats > 1:
                        motion2_output = motion2_output.repeat(num_repeats, 1, 1)

                padding_len = motion1.shape[1] - motion1_output.shape[1]
                if padding_len > 0:
                    B, D = motion1_output.shape[0], motion1_output.shape[2]
                    padding_zeros = torch.zeros((B, padding_len, D)).to(device)
                    motion1_output = torch.concat((motion1_output, padding_zeros), dim=1)
                if motion2_output.shape[1] != motion2.shape[1]:
                    padding_len_2 = motion2.shape[1] - motion2_output.shape[1]
                    if padding_len_2 > 0:
                        B2, D2 = motion2_output.shape[0], motion2_output.shape[2]
                        motion2_output = torch.concat((motion2_output, torch.zeros((B2, padding_len_2, D2)).to(device)), dim=1)

                batch.update({"output": torch.cat([motion1_output, motion2_output], dim=-1)})
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                if opt.save_vis and i == 0:
                    save_vis_n = 20 if trans is not None else 10
                    for b in range(min(motions_output.shape[0], save_vis_n)):
                        motions_input = torch.cat([motion1, motion2], dim=-1)
                        motions_input = motions_input.reshape(motions_input.shape[0], motions_input.shape[1], 2, -1)
                        motions_input = self.normalizer.backward(motions_input.cpu().detach().numpy())

                        preprocess_plot_motion(motions_input[b, :motion_lens[b].item()], text[b],
                                               opt.vis_dir, opt.npy_dir,
                                               f"{file.split('.')[0]}_{b:02d}_gt", foot_ik=False)
                        gen_file_name = f"{file.split('.')[0]}_gtp2_{b:02d}_gen"
                        preprocess_plot_motion(motions_output[b, :motion_lens[b].item()], text[b],
                                               opt.vis_dir, opt.npy_dir,
                                               gen_file_name, foot_ik=True)

                B, T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length

                # Get original batch size (before replication for multimodality)
                original_B = len(text)
                for b in range(original_B):
                    m_len = motion_lens[b]
                    if i in mm_idxs:
                        # For multimodal batches, extract all replications for this prompt
                        mm_sub_dict = {'mm_motions': motions_output[b * num_repeats:(b + 1) * num_repeats, :, :],
                                       'motion_lens': m_len,
                                       'text': text[b]}
                        mm_generated_motions.append(mm_sub_dict)

                        # For generated_motions, take only the first replication
                        sub_dict = {'motion1': motions_output[b * num_repeats, :, 0],
                                    'motion2': motions_output[b * num_repeats, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    else:
                        # No replications, use directly
                        sub_dict = {'motion1': motions_output[b, :, 0],
                                    'motion2': motions_output[b, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    generated_motions.append(sub_dict)

        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


def get_motion_loader_cmdm_gt_p2(batch_size, model, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr, sampler='euler'):
    start = time.time()
    dataset = EvaluationDataset_cmdm_gt_p2(model, trans, ground_truth_dataset, device,
                                           mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats,
                                           file=file, opt=opt, time_steps=time_steps,
                                           cond_scale=cond_scale, topkr=topkr, sampler=sampler)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    print(f'GT-P2 Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


class EvaluationDataset_cmdm_raw(Dataset):
    """Evaluation dataset for DiT models that work directly on raw 262D motion features"""

    def __init__(self, trans, dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
        self.normalizer = MotionNormalizer()
        self.trans = trans.to(device)
        self.trans.eval()

        dataloader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=0, shuffle=True)
        self.max_length = dataset.max_length

        idxs = list(range(len(dataset)))
        mm_idxs = idxs[:mm_num_samples]

        generated_motions = []
        mm_generated_motions = []

        if opt.save_vis:
            print(f"Saving visualizations...")

        with torch.no_grad():
            for i, data in tqdm(enumerate(dataloader)):


                name, text, motion1, motion2, motion_lens = data
                batch = {}
                if i in mm_idxs:
                    num_repeats = mm_num_repeats
                else:
                    num_repeats = 1

                ids_length = motion_lens.detach().long().to(device)

                for num_repeat in range(num_repeats):
                    if opt.gen_react:
                        raise NotImplementedError("Reaction generation is not supported for raw CMDM")
                    else:
                        # Generate motion directly in raw 262D space
                        motion_1_output, motion_2_output = trans.generate(text, ids_length, cond_scale)

                    # Motion is already in raw 262D space, no need to decode
                    motion_1_output = rearrange(motion_1_output, 'b l 1 d -> b l d')
                    motion_2_output = rearrange(motion_2_output, 'b l 1 d -> b l d')

                    if num_repeat == 0:
                        motion1_output = motion_1_output
                        motion2_output = motion_2_output
                    else:
                        motion1_output = torch.cat((motion1_output, motion_1_output), dim=0)
                        motion2_output = torch.cat((motion2_output, motion_2_output), dim=0)

                padding_len = motion1.shape[1] - motion1_output.shape[1]
                B, D = motion1_output.shape[0], motion1_output.shape[2]
                padding_zeros = torch.zeros((B, padding_len, D)).to(device)
                motion1_output = torch.concat((motion1_output, padding_zeros), dim=1)
                motion2_output = torch.concat((motion2_output, padding_zeros), dim=1)

                batch.update({"output": torch.cat([motion1_output, motion2_output], dim=-1)})
                motions_output = batch["output"].reshape(batch["output"].shape[0], batch["output"].shape[1], 2, -1)
                motions_output = self.normalizer.backward(motions_output.cpu().detach().numpy())

                if opt.save_vis and i == 0:
                    save_vis_n = 20

                    for b in range(B):
                        if b >= save_vis_n:
                            break

                        motions_input = torch.cat([motion1, motion2], dim=-1)
                        motions_input = motions_input.reshape(motions_input.shape[0], motions_input.shape[1], 2, -1)
                        motions_input = self.normalizer.backward(motions_input.cpu().detach().numpy())

                        preprocess_plot_motion(motions_input[b, :motion_lens[b].item()],  text[b],
                                            opt.vis_dir, opt.npy_dir,
                                            f"{file.split('.')[0]}_{b:02d}_gt", foot_ik=False)

                        gen_file_name = f"{file.split('.')[0]}_raw_{b:02d}_gen"

                        preprocess_plot_motion(motions_output[b, :motion_lens[b].item()], text[b],
                                            opt.vis_dir, opt.npy_dir,
                                            gen_file_name, foot_ik=True,)

                    raise Exception("Visualization saved")


                B,T = motions_output.shape[0], motions_output.shape[1]
                if T < self.max_length:
                    padding_len = self.max_length - T
                    D = motions_output.shape[-1]
                    padding_zeros = np.zeros((B, padding_len, 2, D))
                    motions_output = np.concatenate((motions_output, padding_zeros), axis=1)
                assert motions_output.shape[1] == self.max_length

                # Get original batch size (before replication for multimodality)
                original_B = len(text)
                for b in range(original_B):
                    m_len = motion_lens[b]
                    if i in mm_idxs:
                        # For multimodal batches, extract all replications for this prompt
                        mm_sub_dict = {'mm_motions': motions_output[b*num_repeats:(b+1)*num_repeats, :, :],
                                       'motion_lens': m_len,
                                       'text': text[b]}
                        mm_generated_motions.append(mm_sub_dict)

                        # For generated_motions, take only the first replication
                        sub_dict = {'motion1': motions_output[b*num_repeats, :, 0],
                                    'motion2': motions_output[b*num_repeats, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    else:
                        # No replications, use directly
                        sub_dict = {'motion1': motions_output[b, :, 0],
                                    'motion2': motions_output[b, :, 1],
                                    'motion_lens': m_len,
                                    'text': text[b]}
                    generated_motions.append(sub_dict)


        self.generated_motions = generated_motions
        self.mm_generated_motions = mm_generated_motions

    def __len__(self):
        return len(self.generated_motions)

    def __getitem__(self, item):
        data = self.generated_motions[item]
        motion1, motion2, motion_lens, text = data['motion1'], data['motion2'], data['motion_lens'], data['text']
        return "generated", text, motion1, motion2, motion_lens


def get_motion_loader_cmdm_raw(batch_size, trans, ground_truth_dataset, device, mm_num_samples, mm_num_repeats, file, opt, time_steps, cond_scale, topkr):
    """Motion loader for DiT models that work directly on raw 262D motion features"""
    start = time.time()
    dataset = EvaluationDataset_cmdm_raw(trans, ground_truth_dataset, device, mm_num_samples=mm_num_samples, mm_num_repeats=mm_num_repeats, file=file, opt=opt, time_steps=time_steps, cond_scale=cond_scale, topkr=topkr)
    mm_dataset = MMGeneratedDataset(dataset)

    motion_loader = DataLoader(dataset, batch_size=batch_size, drop_last=True, num_workers=0, shuffle=True)
    if mm_dataset.dataset:
        mm_motion_loader = DataLoader(mm_dataset, batch_size=1, num_workers=0)
    else:
        mm_motion_loader = None

    print(f'Raw Generated Dataset Loading Completed using {file} in {(time.time() - start) / 60:.2f} min!!!')

    return motion_loader, mm_motion_loader


def build_models(cfg):
    model = InterCLIP(cfg)

    checkpoint = torch.load(pjoin('checkpoints/eval_model/interclip.ckpt'),map_location="cpu", weights_only=False)
    # checkpoint = torch.load(pjoin('checkpoints/interclip/model/5.ckpt'),map_location="cpu")
    for k in list(checkpoint["state_dict"].keys()):
        if "model" in k:
            checkpoint["state_dict"][k.replace("model.", "")] = checkpoint["state_dict"].pop(k)
    model.load_state_dict(checkpoint["state_dict"], strict=True)

    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return model


class EvaluatorModelWrapper(object):

    def __init__(self, cfg, device):

        self.model = build_models(cfg)
        self.cfg = cfg
        self.device = device

        self.model = self.model.to(device)
        self.model.eval()


    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

            '''Text Encoding'''
            text_embedding = self.model.encode_text(batch)['text_emb'][align_idx]

        return text_embedding, motion_embedding

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, batch_data):
        with torch.no_grad():
            name, text, motion1, motion2, motion_lens = batch_data
            motion1 = motion1.detach().float()  # .to(self.device)
            motion2 = motion2.detach().float()  # .to(self.device)
            motions = torch.cat([motion1, motion2], dim=-1)
            motions = motions.detach().to(self.device).float()

            align_idx = np.argsort(motion_lens.data.tolist())[::-1].copy()
            motions = motions[align_idx]
            motion_lens = motion_lens[align_idx]
            text = list(text)

            B, T = motions.shape[:2]
            cur_len = torch.LongTensor([min(T, m_len) for m_len in motion_lens]).to(self.device)
            padded_len = cur_len.max()

            batch = {}
            batch["text"] = text
            batch["motions"] = motions.reshape(B, T, -1)[:, :padded_len]
            batch["motion_lens"] = motion_lens

            '''Motion Encoding'''
            motion_embedding = self.model.encode_motion(batch)['motion_emb']

        return motion_embedding
