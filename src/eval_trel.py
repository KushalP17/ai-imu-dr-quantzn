# eval_trel.py
# Run AFTER test_filter has been executed (results must exist in args.path_results).
# Usage: python eval_trel.py
#
# Computes t_rel (%) and r_rel (deg/km) per sequence and prints a summary table,
# matching the exact metric definition from Brossard et al. (2020).

import os
import numpy as np
import torch
from termcolor import cprint
from main_kitti import KITTIArgs, KITTIDataset, KITTIParameters
from utils import prepare_data


SEQ_LENGTHS = [100, 200, 300, 400, 500, 600, 700, 800]   # metres
STEP_SIZE   = 10   # evaluate a new sub-sequence every 10 samples (at 1 Hz = 10 s)


# ---------------------------------------------------------------------------
# Core metric functions
# ---------------------------------------------------------------------------

def compute_trel_rrel(Rot_est, p_est, Rot_gt, p_gt):
    """
    Compute relative translational error (t_rel, %) and relative rotational
    error (r_rel, deg/km) following the KITTI odometry benchmark definition.

    All tensors are at the original 100 Hz sample rate.
    Rot_*: (N, 3, 3) numpy arrays
    p_*  : (N, 3)    numpy arrays
    """
    # --- downsample to 1 Hz (every 10th sample) ----------------------------
    Rot_gt_1hz  = Rot_gt[::10]
    p_gt_1hz    = p_gt[::10]
    Rot_est_1hz = Rot_est[::10]
    p_est_1hz   = p_est[::10]
    M = p_gt_1hz.shape[0]

    # --- cumulative distance along ground-truth path -----------------------
    dp = p_gt_1hz[1:] - p_gt_1hz[:-1]                  # (M-1, 3)
    distances      = np.zeros(M)
    distances[1:]  = np.linalg.norm(dp, axis=1).cumsum()

    trans_errors = []
    rot_errors   = []

    for i in range(0, M - 1, STEP_SIZE):
        for L in SEQ_LENGTHS:
            # find the index j such that the path from i to j covers L metres
            target = distances[i] + L
            if target > distances[-1]:
                continue
            j = i + int(np.searchsorted(distances[i:], target))
            if j >= M:
                continue

            # ---- translational error --------------------------------------
            # local displacement in the starting body frame (ground truth)
            dp_gt  = Rot_gt_1hz[i].T  @ (p_gt_1hz[j]  - p_gt_1hz[i])
            # local displacement in the starting body frame (estimate)
            dp_est = Rot_est_1hz[i].T @ (p_est_1hz[j] - p_est_1hz[i])
            dist_gt = np.linalg.norm(dp_gt)
            if dist_gt < 1e-6:
                continue
            trans_errors.append(np.linalg.norm(dp_est - dp_gt) / dist_gt)

            # ---- rotational error (deg / km) ------------------------------
            # relative rotation ground truth: R_i^T R_j
            dR_gt  = Rot_gt_1hz[i].T  @ Rot_gt_1hz[j]
            # relative rotation estimate
            dR_est = Rot_est_1hz[i].T @ Rot_est_1hz[j]
            # rotation error matrix
            dR_err = dR_gt.T @ dR_est
            # angle of the error rotation (Rodrigues formula)
            trace  = np.clip((np.trace(dR_err) - 1.0) / 2.0, -1.0, 1.0)
            angle_rad = np.abs(np.arccos(trace))
            # normalise to deg per km
            rot_errors.append(np.degrees(angle_rad) / (dist_gt / 1000.0))

    if len(trans_errors) == 0:
        return None, None

    t_rel = 100.0 * float(np.mean(trans_errors))   # percentage
    r_rel = float(np.mean(rot_errors))              # deg / km
    return t_rel, r_rel


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args, dataset):
    results = {}

    for i in range(len(dataset.datasets)):
        dataset_name = dataset.dataset_name(i)

        # Only evaluate sequences that are part of the odometry benchmark
        if dataset_name not in dataset.odometry_benchmark:
            continue

        # Check that test_filter has already saved estimates for this sequence
        result_file = os.path.join(args.path_results, dataset_name + "_filter.p")
        if not os.path.exists(result_file):
            cprint(f"No saved result for {dataset_name}, skipping.", 'yellow')
            continue

        # Load filter estimates
        mondict    = dataset.load(result_file)
        Rot_est    = mondict['Rot']   # (N, 3, 3) numpy
        p_est      = mondict['p']     # (N, 3)    numpy

        # Load ground truth (benchmark slice only)
        Ns = dataset.odometry_benchmark[dataset_name]
        t, ang_gt, p_gt_torch, v_gt, u = prepare_data(
            args, dataset, dataset_name, i,
            idx_start=Ns[0], idx_end=Ns[1],
            to_numpy=True
        )
        p_gt = p_gt_torch   # already numpy after to_numpy=True

        # Align lengths (estimates may cover a different slice than ground truth)
        N = min(Rot_est.shape[0], p_gt.shape[0])
        Rot_est = Rot_est[:N]
        p_est   = p_est[:N]
        p_gt    = p_gt[:N]

        # Rebuild Rot_gt from euler angles (same convention as the rest of the codebase)
        from utils_torch_filter import TORCHIEKF
        Rot_gt = np.zeros((N, 3, 3))
        for k in range(N):
            Rot_gt[k] = TORCHIEKF.from_rpy(
                torch.tensor(ang_gt[k, 0]),
                torch.tensor(ang_gt[k, 1]),
                torch.tensor(ang_gt[k, 2])
            ).numpy()

        t_rel, r_rel = compute_trel_rrel(Rot_est, p_est, Rot_gt, p_gt)

        if t_rel is None:
            cprint(f"{dataset_name}: not enough data to compute t_rel.", 'yellow')
            continue

        results[dataset_name] = {'t_rel': t_rel, 'r_rel': r_rel}
        print(f"  {dataset_name}   t_rel={t_rel:.2f}%   r_rel={r_rel:.2f} deg/km")

    if not results:
        cprint("No results found. Did you run test_filter first?", 'red')
        return

    # Summary
    t_rels = [v['t_rel'] for v in results.values()]
    r_rels = [v['r_rel'] for v in results.values()]
    print("\n" + "="*60)
    print(f"  Mean t_rel : {np.mean(t_rels):.2f}%   (paper reports ~1.10%)")
    print(f"  Mean r_rel : {np.mean(r_rels):.2f} deg/km")
    print("="*60)
    return results


if __name__ == '__main__':
    args    = KITTIArgs()
    dataset = KITTIDataset(args)
    evaluate(args, dataset)