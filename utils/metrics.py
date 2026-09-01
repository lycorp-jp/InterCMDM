import numpy as np
from scipy import linalg
# emb_scale = 6

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists

def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0)
    else:
        return top_k_mat


def calculate_matching_score(embedding1, embedding2, sum_all=False):
    assert len(embedding1.shape) == 2
    assert embedding1.shape[0] == embedding2.shape[0]
    assert embedding1.shape[1] == embedding2.shape[1]

    dist = linalg.norm(embedding1 - embedding2, axis=1)
    if sum_all:
        return dist.sum(axis=0)
    else:
        return dist



def calculate_activation_statistics(activations, emb_scale):
    """
    Params:
    -- activation: num_samples x dim_feat
    Returns:
    -- mu: dim_feat
    -- sigma: dim_feat x dim_feat
    """
    activations = activations * emb_scale
    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_diversity(activation, diversity_times, emb_scale, divide_by):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    activation = activation * emb_scale
    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm((activation[first_indices] - activation[second_indices])/divide_by, axis=1)
    return dist.mean()


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_multimodality(activation, multimodality_times, emb_scale, divide_by):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    activation = activation * emb_scale
    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm((activation[:, first_dices] - activation[:, second_dices])/divide_by, axis=2)
    return dist.mean()


# ============================================================================
# Long-Horizon Evaluation Metrics
# ============================================================================

def compute_segment_wise_fid(gt_embeddings, gen_embeddings, num_segments=5, emb_scale=6):
    """
    Compute FID scores for each segment of long-horizon sequences.

    Args:
        gt_embeddings: Ground truth motion embeddings per segment, shape (num_segments, num_samples, embedding_dim)
        gen_embeddings: Generated motion embeddings per segment, shape (num_segments, num_samples, embedding_dim)
        num_segments: Number of segments to evaluate (default: 5)
        emb_scale: Embedding scale factor (default: 6 for InterHuman)

    Returns:
        segment_fids: Array of FID scores for each segment, shape (num_segments,)
        mean_fid: Mean FID across all segments
        std_fid: Standard deviation of FID across segments
    """
    assert gt_embeddings.shape[0] == num_segments, f"Expected {num_segments} segments, got {gt_embeddings.shape[0]}"
    assert gen_embeddings.shape[0] == num_segments, f"Expected {num_segments} segments, got {gen_embeddings.shape[0]}"

    segment_fids = []

    for seg_idx in range(num_segments):
        gt_seg = gt_embeddings[seg_idx]  # (num_samples, embedding_dim)
        gen_seg = gen_embeddings[seg_idx]  # (num_samples, embedding_dim)

        # Calculate activation statistics for this segment
        gt_mu, gt_cov = calculate_activation_statistics(gt_seg, emb_scale)
        gen_mu, gen_cov = calculate_activation_statistics(gen_seg, emb_scale)

        # Calculate FID for this segment
        fid = calculate_frechet_distance(gt_mu, gt_cov, gen_mu, gen_cov)
        segment_fids.append(fid)

    segment_fids = np.array(segment_fids)
    mean_fid = np.mean(segment_fids)
    std_fid = np.std(segment_fids)

    return segment_fids, mean_fid, std_fid


def compute_transition_smoothness(motions, num_segments=5, fps=30):
    """
    Compute transition smoothness metrics between segments.
    Measures joint position discontinuity and velocity consistency at segment boundaries.

    Args:
        motions: Motion sequences, shape (num_samples, num_frames, 2, num_features)
                 where 2 represents two persons
        num_segments: Number of segments (default: 5)
        fps: Frame rate (default: 30)

    Returns:
        discontinuity: Average joint position discontinuity at segment boundaries (mm)
        velocity_diff: Average velocity difference at segment boundaries (mm/s)
        per_boundary_discontinuity: Discontinuity at each boundary, shape (num_segments-1,)
        per_boundary_velocity_diff: Velocity difference at each boundary, shape (num_segments-1,)
    """
    num_samples, num_frames, num_persons, num_features = motions.shape
    segment_length = num_frames // num_segments

    # Lists to store metrics at each boundary
    boundary_discontinuities = []
    boundary_velocity_diffs = []

    # Iterate through segment boundaries
    for seg_idx in range(num_segments - 1):
        boundary_frame = (seg_idx + 1) * segment_length

        # Get frames before and after the boundary
        frame_before = motions[:, boundary_frame - 1, :, :]  # (num_samples, 2, num_features)
        frame_after = motions[:, boundary_frame, :, :]  # (num_samples, 2, num_features)

        # Compute position discontinuity (L2 norm across features)
        position_discontinuity = np.linalg.norm(frame_after - frame_before, axis=-1)  # (num_samples, 2)
        avg_discontinuity = position_discontinuity.mean()  # Average across samples and persons
        boundary_discontinuities.append(avg_discontinuity)

        # Compute velocity at boundary (approximate using neighboring frames)
        # Velocity before boundary
        if boundary_frame >= 2:
            velocity_before = (frame_before - motions[:, boundary_frame - 2, :, :]) * fps
        else:
            velocity_before = np.zeros_like(frame_before)

        # Velocity after boundary
        if boundary_frame < num_frames - 1:
            velocity_after = (motions[:, boundary_frame + 1, :, :] - frame_after) * fps
        else:
            velocity_after = np.zeros_like(frame_after)

        # Compute velocity difference
        velocity_diff = np.linalg.norm(velocity_after - velocity_before, axis=-1)  # (num_samples, 2)
        avg_velocity_diff = velocity_diff.mean()
        boundary_velocity_diffs.append(avg_velocity_diff)

    # Convert to arrays
    per_boundary_discontinuity = np.array(boundary_discontinuities)
    per_boundary_velocity_diff = np.array(boundary_velocity_diffs)

    # Overall metrics
    discontinuity = per_boundary_discontinuity.mean()
    velocity_diff = per_boundary_velocity_diff.mean()

    return discontinuity, velocity_diff, per_boundary_discontinuity, per_boundary_velocity_diff


def compute_quality_degradation(segment_metrics, metric_name='fid'):
    """
    Analyze quality degradation across segments.

    Args:
        segment_metrics: Metrics for each segment, shape (num_segments,)
                        For FID, higher is worse. For R-precision, lower is worse.
        metric_name: Name of the metric ('fid', 'mm_dist', 'r_precision', etc.)

    Returns:
        degradation_slope: Linear slope of quality degradation (positive means getting worse for FID)
        degradation_rate: Percentage change from first to last segment
        per_segment_change: Change from previous segment, shape (num_segments-1,)
    """
    num_segments = len(segment_metrics)

    # Compute linear regression slope
    x = np.arange(num_segments)
    slope, _ = np.polyfit(x, segment_metrics, 1)

    # Compute degradation rate (percentage change from first to last)
    if segment_metrics[0] != 0:
        degradation_rate = ((segment_metrics[-1] - segment_metrics[0]) / np.abs(segment_metrics[0])) * 100
    else:
        degradation_rate = 0.0

    # Compute per-segment changes
    per_segment_change = np.diff(segment_metrics)

    return slope, degradation_rate, per_segment_change


def compute_error_accumulation(gt_motions, gen_motions, num_segments=5):
    """
    Track error accumulation across segments for long-horizon generation.

    Args:
        gt_motions: Ground truth motions, shape (num_samples, num_frames, 2, num_features)
        gen_motions: Generated motions, shape (num_samples, num_frames, 2, num_features)
        num_segments: Number of segments (default: 5)

    Returns:
        segment_errors: MPJPE for each segment, shape (num_segments,)
        cumulative_errors: Cumulative error up to each segment, shape (num_segments,)
        error_growth_rate: Rate of error growth across segments
    """
    num_samples, num_frames, num_persons, num_features = gt_motions.shape
    segment_length = num_frames // num_segments

    segment_errors = []

    # Compute error for each segment
    for seg_idx in range(num_segments):
        start_frame = seg_idx * segment_length
        end_frame = (seg_idx + 1) * segment_length

        gt_segment = gt_motions[:, start_frame:end_frame, :, :]
        gen_segment = gen_motions[:, start_frame:end_frame, :, :]

        # Compute Mean Per-Joint Position Error (MPJPE)
        error = np.linalg.norm(gt_segment - gen_segment, axis=-1)  # (num_samples, segment_length, 2)
        mpjpe = error.mean()  # Average across all dimensions
        segment_errors.append(mpjpe)

    segment_errors = np.array(segment_errors)

    # Compute cumulative errors
    cumulative_errors = np.cumsum(segment_errors)

    # Compute error growth rate (linear regression slope)
    x = np.arange(num_segments)
    error_growth_rate, _ = np.polyfit(x, segment_errors, 1)

    return segment_errors, cumulative_errors, error_growth_rate