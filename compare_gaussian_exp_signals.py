# Converted from compare_gaussian_exp_signals.ipynb

"""Matrix similarity experiment comparing Gaussian and exponential signals.

Generates synthetic patient signal matrices (channels x frames) from two
distributions, then compares them using multiple matrix similarity measures
across five scenarios: same/different distributions and same/different
channel counts.
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
from scipy.linalg import orthogonal_procrustes
from itertools import product as iter_product
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1. Signal generation
# ---------------------------------------------------------------------------


def generate_signal(distribution, n_samples=10000, width=1.0, peak_magnitude=5.0):
    """Generate a 1-D signal from a given distribution.

    Args:
        distribution: Either 'gaussian' or 'exponential'.
        n_samples: Number of time samples in the raw signal.
        width: Controls spread. Std-dev for Gaussian, scale (1/lambda)
            for exponential.
        peak_magnitude: Target peak amplitude (signal is rescaled).

    Returns:
        1-D numpy array of shape (n_samples,).
    """
    t = np.linspace(-5 * width, 5 * width, n_samples)

    if distribution == "gaussian":
        signal = np.exp(-0.5 * (t / width) ** 2)
    elif distribution == "exponential":
        signal = np.where(t >= 0, np.exp(-t / width), 0.0)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

    signal = signal / (signal.max() + 1e-12) * peak_magnitude
    return signal


def signal_to_frame_matrix(signal, n_frames):
    """Split a signal into frames and return mean amplitude per frame.

    Args:
        signal: 1-D array of raw signal values.
        n_frames: Number of temporal frames to split into.

    Returns:
        1-D array of shape (n_frames,) with mean amplitude per frame.
    """
    frames = np.array_split(signal, n_frames)
    return np.array([np.mean(f) for f in frames])


def generate_patient_matrix(
    distribution,
    n_channels,
    n_frames,
    width,
    peak_magnitude=5.0,
    magnitude_variation=0.3,
    n_samples=10000,
    rng=None,
):
    """Generate a patient matrix of shape (n_channels, n_frames).

    Each row (channel) gets a different magnitude scaling drawn uniformly
    from [1 - magnitude_variation, 1 + magnitude_variation] * peak_magnitude.

    Args:
        distribution: 'gaussian' or 'exponential'.
        n_channels: Number of rows (channels / electrodes).
        n_frames: Number of columns (temporal frames).
        width: Distribution width parameter.
        peak_magnitude: Base peak amplitude.
        magnitude_variation: Fractional spread of per-channel magnitudes.
        n_samples: Raw signal resolution before framing.
        rng: numpy random Generator for reproducibility.

    Returns:
        2-D numpy array of shape (n_channels, n_frames).
    """
    if rng is None:
        rng = np.random.default_rng()

    low = peak_magnitude * (1 - magnitude_variation)
    high = peak_magnitude * (1 + magnitude_variation)
    channel_magnitudes = rng.uniform(low, high, size=n_channels)

    rows = []
    for mag in channel_magnitudes:
        sig = generate_signal(distribution, n_samples=n_samples, width=width, peak_magnitude=mag)
        noise = rng.normal(0, 0.05 * mag, size=len(sig))
        sig = sig + noise
        row = signal_to_frame_matrix(sig, n_frames)
        rows.append(row)

    return np.array(rows)


# ---------------------------------------------------------------------------
# 2. Similarity measures
# ---------------------------------------------------------------------------


def _resample_matrix(M, target_rows, target_cols):
    """Resample a matrix to target dimensions via linear interpolation.

    Treats row and column indices as normalized [0, 1] grids and
    interpolates to the target shape, preserving the overall profile
    without discarding data.

    Args:
        M: 2-D array of shape (m, n).
        target_rows: Desired number of rows.
        target_cols: Desired number of columns.

    Returns:
        2-D array of shape (target_rows, target_cols).
    """
    from scipy.interpolate import RegularGridInterpolator

    m, n = M.shape
    row_coords = np.linspace(0, 1, m)
    col_coords = np.linspace(0, 1, n)
    interp = RegularGridInterpolator(
        (row_coords, col_coords), M, method="linear", bounds_error=False, fill_value=None
    )
    new_rows = np.linspace(0, 1, target_rows)
    new_cols = np.linspace(0, 1, target_cols)
    grid = np.meshgrid(new_rows, new_cols, indexing="ij")
    points = np.stack([grid[0].ravel(), grid[1].ravel()], axis=-1)
    return interp(points).reshape(target_rows, target_cols)


def _normalize_dimensions(A, B):
    """Resample both matrices to their maximum row and column counts.

    The larger matrix keeps its original shape. The smaller matrix is
    upsampled via interpolation along whichever axis is shorter.

    Args:
        A: 2-D array.
        B: 2-D array.

    Returns:
        Tuple (A_resampled, B_resampled) with matching shapes.
    """
    target_rows = max(A.shape[0], B.shape[0])
    target_cols = max(A.shape[1], B.shape[1])
    A_out = A if A.shape == (target_rows, target_cols) else _resample_matrix(A, target_rows, target_cols)
    B_out = B if B.shape == (target_rows, target_cols) else _resample_matrix(B, target_rows, target_cols)
    return A_out, B_out


def _column_gram(M):
    """Compute the column Gram matrix normalized by number of rows.

    Returns (M^T M) / m, which is a (cols x cols) matrix independent
    of the number of rows. This captures the frame-to-frame covariance
    structure.

    Args:
        M: 2-D array of shape (m, n).

    Returns:
        2-D array of shape (n, n).
    """
    return (M.T @ M) / M.shape[0]


def frobenius_cosine(A, B):
    """Frobenius cosine similarity, dimension-agnostic.

    When shapes differ, resamples the smaller matrix via interpolation
    to match the larger, then computes cosine similarity on the
    flattened vectors.

    Args:
        A: 2-D array of shape (m1, n1).
        B: 2-D array of shape (m2, n2).

    Returns:
        Scalar similarity in [-1, 1].
    """
    A_r, B_r = _normalize_dimensions(A, B)
    return 1.0 - cosine_dist(A_r.ravel(), B_r.ravel())


def rv_coefficient(A, B):
    """RV coefficient using row-normalized column Gram matrices.

    Computes column Gram matrices G = (M^T M) / m for each matrix,
    which are (frames x frames) regardless of channel count. This
    compares the temporal covariance structure normalized by the
    number of channels.

    Args:
        A: 2-D array of shape (m1, n1).
        B: 2-D array of shape (m2, n2).

    Returns:
        Scalar in [0, 1].
    """
    if A.shape[1] != B.shape[1]:
        target_cols = max(A.shape[1], B.shape[1])
        A = A if A.shape[1] == target_cols else _resample_matrix(A, A.shape[0], target_cols)
        B = B if B.shape[1] == target_cols else _resample_matrix(B, B.shape[0], target_cols)

    S1 = _column_gram(A)
    S2 = _column_gram(B)
    numerator = np.trace(S1 @ S2)
    denominator = np.sqrt(np.trace(S1 @ S1) * np.trace(S2 @ S2))
    return numerator / (denominator + 1e-12)


def linear_cka(A, B):
    """Linear CKA using column Gram matrices.

    When row counts differ, uses column Gram matrices (frames x frames)
    normalized by row count, which makes the comparison independent of
    channel count. When row counts match, uses the standard row Gram
    formulation.

    Args:
        A: 2-D array of shape (m1, n1).
        B: 2-D array of shape (m2, n2).

    Returns:
        Scalar in [0, 1].
    """
    same_rows = A.shape[0] == B.shape[0]

    if same_rows:
        n = A.shape[0]
        H = np.eye(n) - np.ones((n, n)) / n
        K = A @ A.T
        L = B @ B.T
    else:
        if A.shape[1] != B.shape[1]:
            target_cols = max(A.shape[1], B.shape[1])
            A = A if A.shape[1] == target_cols else _resample_matrix(A, A.shape[0], target_cols)
            B = B if B.shape[1] == target_cols else _resample_matrix(B, B.shape[0], target_cols)
        n = A.shape[1]
        H = np.eye(n) - np.ones((n, n)) / n
        K = _column_gram(A)
        L = _column_gram(B)

    HKH = H @ K @ H
    HLH = H @ L @ H
    num = np.trace(HKH @ HLH)
    denom = np.sqrt(np.trace(HKH @ HKH) * np.trace(HLH @ HLH))
    return num / (denom + 1e-12)


def procrustes_similarity(A, B):
    """Procrustes similarity with interpolation for shape matching.

    Resamples the smaller matrix to match the larger via interpolation,
    then finds the optimal orthogonal rotation minimizing the
    Frobenius distance.

    Args:
        A: 2-D array of shape (m1, n1).
        B: 2-D array of shape (m2, n2).

    Returns:
        Scalar in [0, 1] where 1 means identical after rotation.
    """
    A_r, B_r = _normalize_dimensions(A, B)

    A_norm = A_r / (np.linalg.norm(A_r, "fro") + 1e-12)
    B_norm = B_r / (np.linalg.norm(B_r, "fro") + 1e-12)

    R, _ = orthogonal_procrustes(A_norm, B_norm)
    diff = np.linalg.norm(A_norm - B_norm @ R, "fro")
    max_diff = np.sqrt(2)
    return 1.0 - diff / max_diff


def principal_angle_similarity(A, B, k=None):
    """Principal angle similarity with shared ambient space.

    When row counts differ, resamples both to the larger row count
    via interpolation so the column subspaces live in the same
    ambient space.

    Args:
        A: 2-D array of shape (m1, p).
        B: 2-D array of shape (m2, q).
        k: Number of principal angles to use. Defaults to
            min(rank_A, rank_B).

    Returns:
        Mean cosine of principal angles, scalar in [0, 1].
    """
    if A.shape[0] != B.shape[0]:
        target_rows = max(A.shape[0], B.shape[0])
        A = A if A.shape[0] == target_rows else _resample_matrix(A, target_rows, A.shape[1])
        B = B if B.shape[0] == target_rows else _resample_matrix(B, target_rows, B.shape[1])

    UA, _, _ = np.linalg.svd(A, full_matrices=False)
    UB, _, _ = np.linalg.svd(B, full_matrices=False)

    if k is None:
        k = min(UA.shape[1], UB.shape[1])
    UA = UA[:, :k]
    UB = UB[:, :k]

    cos_angles = np.linalg.svd(UA.T @ UB, compute_uv=False)
    cos_angles = np.clip(cos_angles, 0, 1)
    return np.mean(cos_angles)


def mean_profile_cosine(A, B):
    """Cosine similarity between row-averaged temporal profiles.

    Averages across channels to get a 1-D profile per patient
    (mean amplitude per frame), then computes cosine similarity.
    When frame counts differ, resamples to the larger count.

    This is the simplest dimension-agnostic measure: it asks
    whether the average temporal shape is similar.

    Args:
        A: 2-D array of shape (m1, n1).
        B: 2-D array of shape (m2, n2).

    Returns:
        Scalar similarity in [-1, 1].
    """
    profile_a = A.mean(axis=0)
    profile_b = B.mean(axis=0)

    if len(profile_a) != len(profile_b):
        target = max(len(profile_a), len(profile_b))
        x_a = np.linspace(0, 1, len(profile_a))
        x_b = np.linspace(0, 1, len(profile_b))
        x_t = np.linspace(0, 1, target)
        profile_a = np.interp(x_t, x_a, profile_a)
        profile_b = np.interp(x_t, x_b, profile_b)

    return 1.0 - cosine_dist(profile_a, profile_b)


# ---------------------------------------------------------------------------
# 3. Comparison helpers
# ---------------------------------------------------------------------------

ALL_MEASURES = {
    "frobenius_cosine": frobenius_cosine,
    "rv_coefficient": rv_coefficient,
    "linear_cka": linear_cka,
    "procrustes": procrustes_similarity,
    "principal_angles": principal_angle_similarity,
    "mean_profile_cosine": mean_profile_cosine,
}


def compare_patients(A, B, label=""):
    """Run all similarity measures between two patient matrices.

    All measures handle arbitrary dimension mismatches via
    interpolation or Gram-matrix normalization. No truncation.

    Args:
        A: 2-D patient matrix (channels x frames).
        B: 2-D patient matrix (channels x frames).
        label: Description of the comparison for printing.

    Returns:
        Dict mapping measure name to similarity score.
    """
    results = {}

    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"  A: {A.shape}  |  B: {B.shape}")
    print(f"{'=' * 70}")

    for name, func in ALL_MEASURES.items():
        val = func(A, B)
        results[name] = val
        print(f"  {name:25s} : {val:.4f}")

    return results


# ---------------------------------------------------------------------------
# 4. Run experiment
# ---------------------------------------------------------------------------


def run_experiment(seed=42):
    """Run the full matrix similarity experiment.

    Generates patients under five scenarios and compares them with
    multiple similarity measures.

    Args:
        seed: Random seed for reproducibility.
    """
    rng = np.random.default_rng(seed)

    # Configuration
    peak = 5.0
    n_frames_options = [6, 8, 10]
    n_channels_small = 80
    n_channels_large = 110
    widths = [0.5, 1.0, 2.0]

    print("\n" + "#" * 70)
    print("#  MATRIX SIMILARITY EXPERIMENT")
    print("#" * 70)

    for width in widths:
        print(f"\n\n{'*' * 70}")
        print(f"*  WIDTH = {width}")
        print(f"{'*' * 70}")

        n_frames = rng.choice(n_frames_options)

        # -- Generate patients --

        # Scenario 1: same distribution (gaussian), same channels
        p1_gauss = generate_patient_matrix(
            "gaussian", n_channels_small, n_frames, width, peak, rng=rng
        )
        p2_gauss = generate_patient_matrix(
            "gaussian", n_channels_small, n_frames, width, peak, rng=rng
        )

        # Scenario 2: same distribution (gaussian), different channels
        p3_gauss_big = generate_patient_matrix(
            "gaussian", n_channels_large, n_frames, width, peak, rng=rng
        )

        # Scenario 3: different distribution, same channels
        p4_exp = generate_patient_matrix(
            "exponential", n_channels_small, n_frames, width, peak, rng=rng
        )

        # Scenario 4: different distribution, different channels
        p5_exp_big = generate_patient_matrix(
            "exponential", n_channels_large, n_frames, width, peak, rng=rng
        )

        # -- Comparisons --

        compare_patients(
            p1_gauss,
            p2_gauss,
            f"SCENARIO 1: Same dist (gauss), same dims ({n_channels_small}ch, {n_frames}fr, w={width})",
        )

        compare_patients(
            p1_gauss,
            p3_gauss_big,
            f"SCENARIO 2: Same dist (gauss), diff channels ({n_channels_small} vs {n_channels_large}, w={width})",
        )

        compare_patients(
            p1_gauss,
            p4_exp,
            f"SCENARIO 3: Diff dist (gauss vs exp), same dims ({n_channels_small}ch, {n_frames}fr, w={width})",
        )

        compare_patients(
            p1_gauss,
            p5_exp_big,
            f"SCENARIO 4: Diff dist (gauss vs exp), diff channels ({n_channels_small} vs {n_channels_large}, w={width})",
        )

        compare_patients(
            p4_exp,
            p5_exp_big,
            f"SCENARIO 5: Same dist (exp), diff channels ({n_channels_small} vs {n_channels_large}, w={width})",
        )

    # -- Summary: width effect on cross-distribution similarity --
    print(f"\n\n{'#' * 70}")
    print("#  WIDTH EFFECT ON GAUSSIAN vs EXPONENTIAL SIMILARITY")
    print(f"{'#' * 70}")
    print(f"\n{'Width':<8} {'Frob Cos':<12} {'RV Coeff':<12} {'CKA':<12} {'Procrust':<12} {'Princ Ang':<12} {'Prof Cos':<12}")
    print("-" * 80)

    for width in [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]:
        n_frames = 8
        n_ch = 80
        a = generate_patient_matrix("gaussian", n_ch, n_frames, width, peak, rng=rng)
        b = generate_patient_matrix("exponential", n_ch, n_frames, width, peak, rng=rng)

        row = f"{width:<8.1f}"
        row += f" {frobenius_cosine(a, b):<12.4f}"
        row += f" {rv_coefficient(a, b):<12.4f}"
        row += f" {linear_cka(a, b):<12.4f}"
        row += f" {procrustes_similarity(a, b):<12.4f}"
        row += f" {principal_angle_similarity(a, b):<12.4f}"
        row += f" {mean_profile_cosine(a, b):<12.4f}"
        print(row)


if __name__ == "__main__":
    run_experiment()
