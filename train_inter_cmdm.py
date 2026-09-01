import os
import torch
import numpy as np
import wandb

from torch.utils.data import DataLoader
from os.path import join as pjoin

from models.cmdm.DiT_inter_cmdm import dit
from models.cmdm.transformer_trainer import DiTTransformerTrainer
from models.cmdm.TAE import VAE
from options.trans_option import TrainTransOptions

from utils.get_opt import get_opt
from utils.utils import fixseed, namespace_to_dict


def load_vq_model():
    opt_path = pjoin(opt.checkpoints_dir, opt.dataset_name, opt.vq_name, 'opt.txt')
    vq_opt = get_opt(opt_path, opt.device)

    vq_opt.output_emb_width = 64
    vq_opt.tae_hidden_size = 1024
    vq_opt.down_t = 2
    vq_opt.stride_t = 2
    vq_opt.tae_width = vq_opt.tae_hidden_size
    vq_opt.tae_depth = 3
    vq_opt.dilation_growth_rate = 3
    vq_opt.vq_act = "relu"
    vq_opt.vq_norm = None

    tae_model = VAE(
        input_width=vq_opt.feature_dim,
        output_emb_width=vq_opt.output_emb_width,
        hidden_size=vq_opt.tae_hidden_size,
        down_t=vq_opt.down_t,
        stride_t=vq_opt.stride_t,
        width=vq_opt.tae_width,
        depth=vq_opt.tae_depth,
        dilation_growth_rate=vq_opt.dilation_growth_rate,
        activation=vq_opt.vq_act,
        norm=vq_opt.vq_norm,
        clip_range=[-30, 20],
    )
    ckpt = torch.load(pjoin(vq_opt.checkpoints_dir, vq_opt.dataset_name, vq_opt.name, 'model', 'latest.tar'),  map_location='cpu')
    model_key = 'tae_model' if 'tae_model' in ckpt else 'net'

    missing_keys, unexpected_keys = tae_model.load_state_dict(ckpt[model_key], strict=False)
    assert len(unexpected_keys) == 0
    assert all([k.startswith('decoder.conv') or k.startswith('decoder.resnets')for k in missing_keys])
    print(f'Loading TAE Model {opt.vq_name}, epoch {ckpt["ep"]}')
    return tae_model, vq_opt

if __name__ == '__main__':
    parser = TrainTransOptions()
    opt = parser.parse()
    fixseed(opt.seed)

    opt.device = torch.device("cpu" if opt.gpu_id == -1 else "cuda:" + str(opt.gpu_id))

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

    vq_model, vq_opt = load_vq_model()
    opt.feature_dim = vq_opt.feature_dim

    if opt.dataset_name == "interhuman":
        opt.data_root = 'data/InterHuman'
        opt.joints_num = 22
        opt.dim_joint = 12
        opt.max_motion_length = 300
        opt.test_batch_size = 96
        fps = 30

        from data.interhuman import InterHumanDataset
        from models.evaluator.evaluator import EvaluatorModelWrapper

        opt.mode = "train"
        train_dataset = InterHumanDataset(opt)
        opt.mode = "val"
        val_dataset = InterHumanDataset(opt)

        if opt.do_eval:
            opt.mode = "val"
            test_dataset = InterHumanDataset(opt)
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size, drop_last=True, num_workers=0, shuffle=False)

            evalmodel_cfg = get_opt("checkpoints/eval_model/eval_model.yaml", opt.device, complete=False)
            eval_wrapper = EvaluatorModelWrapper(evalmodel_cfg, opt.device)
        else:
            test_loader = None
            eval_wrapper = None

    elif opt.dataset_name == "interx":
        opt.data_root = 'data/Inter-X_Dataset'
        opt.motion_dir = pjoin(opt.data_root, 'processed/motions')
        opt.text_dir = pjoin(opt.data_root, 'processed/texts_processed')

        opt.motion_rep = "smpl"
        opt.joints_num = 56
        opt.max_motion_length = 156
        opt.max_text_len = 35
        opt.unit_length = 4

        opt.test_batch_size = 32
        vq_opt.dim_joint = 6
        fps = 30

        from data.interx import Text2MotionDatasetV2HHI, collate_fn
        from models.evaluator.evaluator_interx import EvaluatorModelWrapper
        from utils.word_vectorizer import WordVectorizer

        w_vectorizer = WordVectorizer(pjoin(opt.data_root, 'processed/glove'), 'hhi_vab')
        train_dataset = Text2MotionDatasetV2HHI(opt,
                                           pjoin(opt.data_root, 'splits/train.txt'),
                                           w_vectorizer,
                                           pjoin(opt.motion_dir, 'train.h5'),
                                           normalize=True)
        val_dataset = Text2MotionDatasetV2HHI(opt,
                                         pjoin(opt.data_root, 'splits/val.txt'),
                                         w_vectorizer,
                                         pjoin(opt.motion_dir, 'val.h5'),
                                         normalize=True)

        if opt.do_eval:
            test_dataset = Text2MotionDatasetV2HHI(opt,
                                                pjoin(opt.data_root, 'splits/val.txt'),
                                                w_vectorizer,
                                                pjoin(opt.motion_dir, 'val.h5'))
            test_loader = DataLoader(test_dataset, batch_size=opt.test_batch_size,
                                    num_workers=4, drop_last=True, collate_fn=collate_fn, shuffle=True)

            wrapper_opt = get_opt("checkpoints/hhi/Comp_v6_KLD01/opt.txt", opt.device, complete=False)
            eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
        else:
            test_loader = None
            eval_wrapper = None

    else:
        raise KeyError('Dataset Does not Exists')

    # Initialize the multi-modal DiT model
    print('Initializing Multi-Modal DiT with TwoPersonDoubleStreamBlock...')
    mask_transformer = dit(
        input_dim=vq_model.output_emb_width,
        cond_mode='text',
        is_causal=True,
        num_frame_per_block=opt.num_frame_per_block,
    )

    pc_transformer = sum(param.numel() for param in mask_transformer.parameters_wo_clip())
    print('Total parameters of the Multi-Modal DiT: {:.2f}M'.format(pc_transformer / 1000_000))



    train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                              shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=opt.batch_size, drop_last=True, num_workers=4,
                            shuffle=False, pin_memory=True)


    opt.save_vis = False
    opt.gen_react = False

    trainer = DiTTransformerTrainer(opt, mask_transformer, vq_model)

    try:
        trainer.train(train_loader, val_loader, test_loader, eval_wrapper=eval_wrapper)
    finally:
        if wandb_run is not None:
            wandb.finish()
