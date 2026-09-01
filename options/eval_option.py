import argparse
import os
import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def arg_parse():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--gpu_id", type=int, default=0, help='GPU id')
    parser.add_argument('--use_trans', type=str2bool, default=True, help='use transformer')
    parser.add_argument('--name', type=str, default='trans_default', help='Name of the experiment')
    parser.add_argument('--dataset_name', type=str, default='interhuman', help='dataset directory')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
    parser.add_argument('--which_epoch', type=str, default="latest", help='Checkpoint you want to use, {latest, best_fid, etc}')
    parser.add_argument('--batch_size', type=int, default=96, help='Batch size')
    parser.add_argument('--save_vis', type=str2bool, default=False, help='save visualization')

    parser.add_argument('--gen_react', action='store_true', help='generate reaction')
    parser.add_argument('--gt_person1', action='store_true', help='fix Person 1 as ground truth, only generate Person 2')
    parser.add_argument('--gt_person2', action='store_true', help='fix Person 2 as ground truth, only generate Person 1')

    ## attention mask settings for inference
    parser.add_argument('--attention_mask_type', type=str, default=None,
                        choices=[None, 'simultaneous', 'independent', 'reaction', 'leader_follower'],
                        help='Attention mask type for inference. If None, uses model default.')
    parser.add_argument('--reactor_first', type=str2bool, default=False,
                        help='For reaction mask: if True, Person 1 is reactor; if False, Person 2 is reactor')
    parser.add_argument('--lag_frames', type=int, default=3,
                        help='For leader_follower mask: temporal lag in frames')

    ## eval settings
    parser.add_argument('--mm_num_samples', type=int, default=0, help='Number of samples for multimodal evaluation') #100
    parser.add_argument('--mm_num_repeats', type=int, default=30, help='Number of repeats for multimodal evaluation')
    parser.add_argument('--mm_num_times', type=int, default=10, help='Number of times for multimodal evaluation')

    parser.add_argument('--diversity_times', type=int, default=300, help='Number of times for diversity evaluation')
    parser.add_argument('--replication_times', type=int, default=10, help='Number of times for replication evaluation') #20

    ## Transformer sampling
    parser.add_argument("--cond_scales", default=[2], nargs="+", type=float,
                                help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    parser.add_argument("--temperature", default=1., type=float,
                                help="Sampling Temperature.")
    parser.add_argument("--topkr", default=[0.9], type=float,
                                help="Filter out percentil low prop entries.")
    parser.add_argument("--time_steps", default=[20], nargs="+", type=int,
                                help="Mask Generate steps.")
    parser.add_argument('--gumbel_sample', action="store_true", help='True: gumbel sampling, False: categorical sampling.')
    parser.add_argument("--sampler", type=str, default="euler", choices=["euler", "heun2"],
                                help="Sampling method: 'euler' (1st order) or 'heun2' (2nd order, more accurate but 2x slower).")
    parser.add_argument('--sigma_min', type=float, default=0.05, help='sigma min')

    opt = parser.parse_args()
    torch.cuda.set_device(opt.gpu_id)
    return opt