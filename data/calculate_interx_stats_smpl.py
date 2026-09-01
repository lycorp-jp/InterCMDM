"""
Calculate mean and std statistics for Inter-X dataset (SMPL representation)
This script processes the Inter-X dataset using SMPL representation format with velocity.
"""

import argparse
import h5py
import numpy as np
import torch
from tqdm import tqdm
import codecs as cs
from os.path import join as pjoin
import os

import data.rotation_conversions as geometry


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))
    return ndarray


def to_rot_6d_smpl(pose, num_person=2):
    """
    Convert axis-angle rotation to 6D rotation representation for SMPL format
    Args:
        pose: Tensor of shape (T, J, D) - axis-angle rotations
        num_person: Number of persons (1 or 2)
    Returns:
        Tensor of shape (T, J, 6*num_rotations) - 6D rotation representation
    """
    pose = to_torch(pose)

    pose_all = []
    # First rotation (root or person 1 root)
    pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:, :, 0:3])))

    # Second rotation (person 2 root) - only for two-person interactions
    if num_person == 2:
        pose_all.append(geometry.matrix_to_rotation_6d(geometry.axis_angle_to_matrix(pose[:, :, 3:6])))

    ret = torch.cat(pose_all, dim=2)
    return ret


def process_motion_smpl(motion, num_person=2):
    """
    Process motion data into SMPL representation format with velocity
    Args:
        motion: Array of shape (T, J, D)
        num_person: Number of persons
    Returns:
        motion_processed: Processed motion tensor (T, J, D_out)
    """
    motion = to_torch(motion)

    # Split rotations and translation
    # All joints except last contain rotations
    # Last joint contains translation for all persons
    rot_6d = to_rot_6d_smpl(motion[:, :-1, :], num_person=num_person).float()

    transl = to_torch(motion[:, -1, :])  # (T, 3*num_person)

    # Compute velocity
    vel = transl[1:] - transl[:-1]
    # Pad last frame with zeros to maintain sequence length
    vel = torch.cat([vel, torch.zeros(1, vel.shape[-1])], axis=0)

    # Concatenate translation and velocity for each person
    transl_vel_all = []
    for ii in range(num_person):
        transl_vel_all.append(
            torch.cat([transl[:, 3*ii:3*ii+3], vel[:, 3*ii:3*ii+3]], axis=1)
        )
    transl_vel_all = torch.cat(transl_vel_all, axis=1)  # (T, 6*num_person)

    # Concatenate rotations and translation/velocity
    # rot_6d: (T, J-1, D_rot), transl_vel_all: (T, 6*num_person)
    # Add joint dimension to transl_vel_all: (T, 1, 6*num_person)
    motion_processed = torch.cat([rot_6d, transl_vel_all.unsqueeze(1)], axis=1)

    return motion_processed


def calculate_stats(args):
    """
    Calculate mean and std statistics for Inter-X dataset (SMPL format)
    """
    print(f"Loading Inter-X dataset from: {args.motion_file}")
    print(f"Using split file: {args.split_file}")
    print(f"Motion representation: SMPL with velocity")

    # Ensure we're using the non-global h5 file
    if args.motion_file.endswith('_global.h5'):
        base_file = args.motion_file.replace('_global.h5', '.h5')
        if os.path.exists(base_file):
            print(f"Using SMPL representation file: {base_file}")
            args.motion_file = base_file
        else:
            print(f"Warning: SMPL file not found at {base_file}")

    # Load ID list
    id_list = []
    with cs.open(args.split_file, 'r') as f:
        for line in f.readlines():
            id_list.append(line.strip())

    print(f"Found {len(id_list)} sequences in split file")

    # Collect all motion data
    all_motions = []

    with h5py.File(args.motion_file, 'r') as mf:
        print("Processing motion sequences...")
        for name in tqdm(id_list):
            try:
                motion = mf[name][:].astype('float32')

                # Skip sequences that are too short
                if motion.shape[0] < args.min_motion_len:
                    continue

                # Process motion into SMPL representation with velocity
                motion_processed = process_motion_smpl(motion, num_person=args.num_person)

                T, J, D = motion_processed.shape
                D = D//2
                all_motions.append(motion_processed[:, :, :D].numpy())
                all_motions.append(motion_processed[:, :, D:].numpy())

            except Exception as e:
                print(f"Error processing {name}: {e}")
                continue

    if len(all_motions) == 0:
        raise ValueError("No valid motion sequences found!")

    print(f"\nSuccessfully processed {len(all_motions)} sequences")

    # Concatenate all data
    all_motions = np.concatenate(all_motions, axis=0)
    print(f"Total data shape: {all_motions.shape}")

    # Calculate statistics
    mean = np.mean(all_motions, axis=0)
    std = np.std(all_motions, axis=0)

    # Avoid division by zero
    std = np.where(std < 1e-6, 1.0, std)

    print("\n=== SMPL Representation Statistics ===")
    print(f"Total data points: {all_motions.shape[0]}")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean range: [{mean.min():.6f}, {mean.max():.6f}]")
    print(f"Std range: [{std.min():.6f}, {std.max():.6f}]")
    print(f"Feature dimension: {mean.shape[0]}D")

    # Save statistics
    os.makedirs(args.output_dir, exist_ok=True)

    output_prefix = f"interx_smpl"
    if args.num_person == 1:
        output_prefix += "_single"

    np.save(pjoin(args.output_dir, f'{output_prefix}_mean.npy'), mean)
    np.save(pjoin(args.output_dir, f'{output_prefix}_std.npy'), std)

    print(f"\nSaved statistics to: {args.output_dir}")
    print(f"  - {output_prefix}_mean.npy")
    print(f"  - {output_prefix}_std.npy")

    # Analyze the structure
    print("\n=== Data Structure Analysis ===")
    print(f"Number of persons: {args.num_person}")

    # Read one sample to get dimensions
    with h5py.File(args.motion_file, 'r') as mf:
        sample_motion = mf[id_list[0]][:].astype('float32')
        processed = process_motion_smpl(sample_motion, num_person=args.num_person)
        T, J, D = processed.shape
        print(f"Sample processed shape: (T={T}, J={J}, D={D})")
        print(f"  - J-1={J-1} joints with rotations (6D per person: {(J-1)*D} dims)")
        print(f"  - 1 joint with translation+velocity ({6*args.num_person} dims)")
        print(f"Total feature dimension after flattening: {J*D}D")


def main():
    parser = argparse.ArgumentParser(description='Calculate Inter-X SMPL representation statistics')
    parser.add_argument('--motion_file', type=str,
                       default='./data/Inter-X/motions_smpl.h5',
                       help='Path to Inter-X motion h5 file (SMPL format, not _global)')
    parser.add_argument('--split_file', type=str,
                       default='./data/Inter-X/split/train.txt',
                       help='Path to train split file')
    parser.add_argument('--output_dir', type=str,
                       default='./data/stats/',
                       help='Directory to save statistics')
    parser.add_argument('--min_motion_len', type=int, default=24,
                       help='Minimum motion sequence length')
    parser.add_argument('--num_person', type=int, default=2,
                       help='Number of persons (1 or 2)')

    args = parser.parse_args()

    calculate_stats(args)


if __name__ == '__main__':
    main()
