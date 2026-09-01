import torch
import numpy as np
import torch
from einops import einsum, parse_shape, repeat
import numpy as np
from einops import rearrange, repeat


def extract_coordinate(
    motion: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    if isinstance(motion, np.ndarray):

        def stack_fn(x):
            return np.stack(x, axis=-1)

        def dim_fn(x):
            return x.ndim

        y_axis = np.array([0, 1, 0], dtype=np.float32)
        mask_y = np.array([1, 0, 1], dtype=np.float32)

        def norm_fn(x):
            return np.clip(np.linalg.norm(x, axis=-1, keepdims=True), a_min=1e-8, a_max=None)

        def cross_fn(x, y):
            return np.cross(x, y, axis=-1)
    else:

        def stack_fn(x):
            return torch.stack(x, dim=-1)

        def dim_fn(x):
            return x.dim()

        y_axis = torch.tensor([0, 1, 0], dtype=torch.float32, device=motion.device)
        mask_y = torch.tensor([1, 0, 1], dtype=torch.float32, device=motion.device)

        def norm_fn(x):
            return torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=1e-8)

        def cross_fn(x, y):
            return torch.cross(x, y, dim=-1)

    if dim_fn(motion) == 3:
        B, T, _ = motion.shape
        output_shape = (
            B,
            T,
        )
    elif dim_fn(motion) == 2:
        T, _ = motion.shape
        motion = rearrange(motion, "n d -> 1 n d")
        B = 1
        output_shape = (T,)
    elif dim_fn(motion) == 1:
        motion = rearrange(motion, "d -> 1 1 d")
        B, T = 1, 1
        output_shape = ()
    else:
        raise ValueError(f"Invalid motion shape: {motion.shape}")

    lhip_id, rhip_id = 1, 2
    joints = motion[..., : 22 * 3].reshape(B, T, 22, 3)
    x_axis = joints[:, :, lhip_id] - joints[:, :, rhip_id]
    x_axis = x_axis * repeat(mask_y, "d -> b t d", b=B, t=T)
    x_axis = x_axis / norm_fn(x_axis)
    y_axis = repeat(y_axis, "d -> b t d", b=B, t=T)
    z_axis = cross_fn(x_axis, y_axis)
    z_axis = z_axis / norm_fn(z_axis)
    R = stack_fn([x_axis, y_axis, z_axis])
    t = joints[:, :, 0]
    t = t * repeat(mask_y, "d -> b t d", b=B, t=T)
    return R.reshape(output_shape + (3, 3)), t.reshape(output_shape + (3,))


def reset_coordinate(
    motion: np.ndarray | torch.Tensor,
    motion2: np.ndarray | torch.Tensor | None = None,
    reference: np.ndarray | torch.Tensor | None = None,
) -> np.ndarray | torch.Tensor:
    shape = motion.shape
    if len(shape) == 3:  # batch
        motion_shape = "b t"
        R_shape = "b"
        t_shape = "b"
    elif len(shape) == 2:  # sequence
        motion_shape = "t"
        R_shape = ""
        t_shape = ""

    if reference is None:
        if len(shape) == 3:
            R, t = extract_coordinate(motion[:, 0])
        else:
            R, t = extract_coordinate(motion[0])
        R_inv, t_inv = inv_transform(R, t, R_shape, t_shape)
        if motion2 is None:
            return rigid_transform(motion, R_inv, t_inv, motion_shape, R_shape, t_shape)
        else:
            return rigid_transform(
                motion, R_inv, t_inv, motion_shape, R_shape, t_shape
            ), rigid_transform(motion2, R_inv, t_inv, motion_shape, R_shape, t_shape)
    else:
        if len(shape) == 3:
            R, t = extract_coordinate(reference[:, 0])
        else:
            R, t = extract_coordinate(reference[0])
        return rigid_transform(motion, R, t, motion_shape, R_shape, t_shape)


def extract_transformation(motion1: np.ndarray | torch.Tensor, motion2: np.ndarray | torch.Tensor, motion_shape="", R_shape="", t_shape=""):
    """
    both motion1 and motion2 are in the motion1's coordinate system, that means, motion1 is in the zero coordinate

    motion_2 = rigid_transform(motion2_origin, R_1_to_2, t_1_to_2, "t", "", "")
    """
    R1, t1 = extract_coordinate(motion1)
    R2, t2 = extract_coordinate(motion2)

    if isinstance(R1, np.ndarray):
        R_1_to_2 = np.matmul(R1.transpose(0, 2, 1), R2)
        t_1_to_2 = matvec(R1.transpose(0, 2, 1), (t2 - t1), R_shape, t_shape)
    else:
        R_1_to_2 = torch.matmul(R1.transpose(2, 1), R2)
        t_1_to_2 = matvec(R1.transpose(2, 1), (t2 - t1), R_shape, t_shape)
    return R_1_to_2, t_1_to_2


def matvec(matrix, vector, matrix_shape="", vector_shape=""):
    """
    Flexible matrix-vector multiplication with automatic broadcasting.
    Supports both NumPy and PyTorch tensors.

    Parameters
    ----------
    :param matrix: array_like
    :param vector: array_like
    :param matrix_shape: str, e.g. "" "t" "b t"
    :param vector_shape: str, e.g. "" "t" "b t"

    Returns
    -------
    :return: array_like
    """
    mat_shape = matrix.shape
    vec_shape = vector.shape
    assert len(mat_shape) >= 2, f"Matrix must be at least 2D, got shape {mat_shape}"
    assert len(vec_shape) >= 1, f"Vector must be at least 1D, got shape {vec_shape}"
    assert vec_shape[-1] == mat_shape[-1], (
        f"Vector last dim {vec_shape[-1]} must match matrix last dim {mat_shape[-1]}"
    )
    result = einsum(matrix, vector, f"{matrix_shape} m n, {vector_shape} n -> {vector_shape} m")
    return result


def inv_transform(R, t, R_shape="", t_shape=""):
    if isinstance(R, np.ndarray):
        shape = tuple(list(range(len(R.shape) - 2))) + (-1, -2)
        R_inv = np.transpose(R, shape)
        t_inv = matvec(R_inv, -t, R_shape, t_shape)
    elif isinstance(R, torch.Tensor):
        R_inv = R.transpose(-1, -2)
        t_inv = matvec(R_inv, -t, R_shape, t_shape)
    else:
        raise ValueError(f"Unsupported type: {type(R)}")
    return R_inv, t_inv


def rigid_transform(
    motion: np.ndarray | torch.Tensor,
    R: np.ndarray | torch.Tensor,
    t: np.ndarray | torch.Tensor,
    motion_shape: str,  # "" "t" "b t"
    R_shape: str,
    t_shape: str,
) -> np.ndarray | torch.Tensor:
    if isinstance(motion, np.ndarray):

        def concat_fn(x, dim=-1):
            return np.concatenate(x, axis=dim)
    else:

        def concat_fn(x, dim=-1):
            return torch.cat(x, dim=dim)

    motion_shape = f"{motion_shape} j"

    shape = motion.shape[:-1]
    joints = motion[..., : 22 * 3].reshape(shape + (22, 3))
    joints_transformed = matvec(R, joints, R_shape, motion_shape) + repeat(
        t, f"{t_shape} n -> {motion_shape} n", **(parse_shape(joints, f"{motion_shape} n"))
    )
    joints_transformed = joints_transformed.reshape(shape + (22 * 3,))
    if motion.shape[-1] == 66:
        return joints_transformed

    else:
        joint_vels = motion[..., 22 * 3 : 22 * 6].reshape(shape + (22, 3))
        rest = motion[..., 22 * 6 :]
        joint_vels_transformed = matvec(R, joint_vels, R_shape, motion_shape)
        joint_vels_transformed = joint_vels_transformed.reshape(shape + (22 * 3,))
        return concat_fn([joints_transformed, joint_vels_transformed, rest], dim=-1)
