import os
from os.path import join as pjoin

import torch
from torch.utils.data import DataLoader
import wandb

from models.cmdm.TAE import VAE
from models.cmdm.tae_trainer import TAETrainer
from options.vq_option import arg_parse
from utils.get_opt import get_opt
from utils.utils import namespace_to_dict

os.environ["OMP_NUM_THREADS"] = "1"

if __name__ == "__main__":
    opt = arg_parse(True)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))
    print(f"Using Device: {opt.device}")

    opt.save_root = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.name)
    opt.model_dir = pjoin(opt.save_root, 'model')
    opt.anim_dir = pjoin(opt.save_root, 'animation')
    opt.eval_dir = pjoin(opt.save_root, 'eval')
    opt.log_dir = pjoin(opt.save_root, 'log')

    os.makedirs(opt.model_dir, exist_ok=True)
    os.makedirs(opt.anim_dir, exist_ok=True)
    os.makedirs(opt.eval_dir, exist_ok=True)
    os.makedirs(opt.log_dir, exist_ok=True)

    wandb_kwargs = dict(
        name=opt.name,
        project=opt.project_name,
        entity=opt.entity,
        config=namespace_to_dict(opt),
    )
    if opt.entity == "user_name":
        wandb_kwargs["mode"] = "disabled"
    wandb_run = wandb.init(**wandb_kwargs)

    test_loader = None
    eval_wrapper = None

    if opt.dataset_name == "interhuman":
        opt.data_root = 'data/InterHuman'
        opt.joints_num = 22
        opt.dim_joint = 12
        opt.test_batch_size = 96

        from data.interhuman import InterHumanMotion

        opt.mode = "train"
        train_dataset = InterHumanMotion(opt)
        opt.mode = "val"
        val_dataset = InterHumanMotion(opt)

    elif opt.dataset_name == "interx":
        opt.data_root = 'data/Inter-X_Dataset'
        opt.motion_dir = pjoin(opt.data_root, 'processed/motions')
        opt.motion_rep = "smpl"
        opt.joints_num = 56
        opt.dim_joint = 6
        opt.max_motion_length = 150
        opt.max_text_len = 35
        opt.unit_length = 4
        opt.test_batch_size = 32

        from data.interx import MotionDatasetV2HHI

        train_dataset = MotionDatasetV2HHI(opt,
                                           pjoin(opt.data_root, 'splits/train.txt'),
                                           pjoin(opt.motion_dir, 'train.h5'),
                                           normalize=True)
        val_dataset = MotionDatasetV2HHI(opt,
                                         pjoin(opt.data_root, 'splits/val.txt'),
                                         pjoin(opt.motion_dir, 'val.h5'),
                                         normalize=True)

    else:
        raise KeyError('Dataset Does not Exists')

    if opt.dataset_name == "interhuman":
        opt.feature_dim = 262
        root_dim = 66
    elif opt.dataset_name == "interx":
        opt.feature_dim = 336
        root_dim = 0
    else:
        raise KeyError('Dataset Does not Exists')

    opt.output_emb_width = 64
    opt.tae_hidden_size = 1024
    opt.down_t = 2
    opt.stride_t = 2
    opt.tae_width = opt.tae_hidden_size
    opt.tae_depth = 3
    opt.dilation_growth_rate = 3
    opt.vq_act = "relu"
    opt.vq_norm = None

    tae_model = VAE(
        input_width=opt.feature_dim,
        output_emb_width=opt.output_emb_width,
        hidden_size=opt.tae_hidden_size,
        down_t=opt.down_t,
        stride_t=opt.stride_t,
        width=opt.tae_width,
        depth=opt.tae_depth,
        dilation_growth_rate=opt.dilation_growth_rate,
        activation=opt.vq_act,
        norm=opt.vq_norm,
        clip_range=[-30, 20],
        root_dim=-1,
    )

    print ("Model Parameters:")
    print ("output_emb_width: ", opt.output_emb_width)
    print ("hidden_size: ", opt.tae_hidden_size)
    print ("down_t: ", opt.down_t)
    print ("stride_t: ", opt.stride_t)
    print ("width: ", opt.tae_width)
    print ("depth: ", opt.tae_depth)
    print ("dilation_growth_rate: ", opt.dilation_growth_rate)
    print ("activation: ", opt.vq_act)
    print ("norm: ", opt.vq_norm)

    total_params = sum(param.numel() for param in tae_model.parameters())
    print(tae_model)
    print('Total parameters of TAE: {:.3f}M'.format(total_params / 1_000_000))

    trainer = TAETrainer(opt, tae_model=tae_model)

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt.batch_size,
        drop_last=True,
        num_workers=4,
        shuffle=False,
        pin_memory=True,
    )

    try:
        trainer.train(train_loader, val_loader, test_loader=test_loader, eval_wrapper=eval_wrapper)
    finally:
        if wandb_run is not None:
            wandb.finish()