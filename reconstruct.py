# reconstruct.py
"""
Improved jumbled-frame reconstructor.
Features:
 - Hybrid similarity: SSIM + color histogram correlation + ORB feature matching
 - Bidirectional greedy ordering from a robust anchor frame
 - Multiprocessing for pairwise similarity computation
 - 2-opt refinement to fix local errors
 - Outputs reconstructed video + run_log.txt + extracted frames/

Usage:
    python reconstruct.py --input jumbled_video.mp4 --output reconstructed_best.mp4 --fps 60 --workers 6
"""

import os
import time
import argparse
from functools import partial
from pathlib import Path
from collections import deque
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

# -------------------------
# Helpers: feature extraction
# -------------------------
def extract_frames(video_path, out_dir):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    idx = 0
    out_dir.mkdir(exist_ok=True)
    success, frame = cap.read()
    while success:
        fname = out_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(fname), frame)
        frames.append(str(fname))
        idx += 1
        success, frame = cap.read()
    cap.release()
    return frames, fps

def load_image_for_features(path, resize=(256,256)):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read {path}")
    img = cv2.resize(img_bgr, resize, interpolation=cv2.INTER_AREA)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_gray

def color_histogram(img_bgr, bins=(8,8,8)):
    # compute normalized 3D color histogram over BGR
    hist = cv2.calcHist([img_bgr], [0,1,2], None, bins, [0,256,0,256,0,256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def orb_descriptors(img_gray, nfeatures=500):
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kp, des = orb.detectAndCompute(img_gray, None)
    return des

def orb_match_score(des1, des2, ratio=0.75):
    if des1 is None or des2 is None:
        return 0.0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    try:
        matches = bf.knnMatch(des1, des2, k=2)
    except Exception:
        return 0.0
    good = 0
    for m_n in matches:
        if len(m_n) != 2:
            continue
        m, n = m_n
        if m.distance < ratio * n.distance:
            good += 1
    denom = max(1, min(len(des1), len(des2)))
    return good / denom

# -------------------------
# Pairwise similarity (composite)
# -------------------------
def pair_similarity(i_j, gray_images_resized, color_hists, descriptors, resize_for_ssim=(256,256)):
    i, j = i_j
    # SSIM (grayscale) - returns in [-1,1], typical [0,1]
    img_i = gray_images_resized[i]
    img_j = gray_images_resized[j]
    try:
        ssim_val = ssim(img_i, img_j, data_range=img_j.max() - img_j.min())
    except Exception:
        # fallback small value if SSIM fails
        ssim_val = 0.0
    # Color histogram correlation (cv2.compareHist returns [-1..1] for some metrics)
    hist_i = color_hists[i]
    hist_j = color_hists[j]
    # use correlation; shift to [0,1]
    try:
        hist_corr = cv2.compareHist(hist_i.astype('float32'), hist_j.astype('float32'), cv2.HISTCMP_CORREL)
    except Exception:
        hist_corr = 0.0
    hist_score = (hist_corr + 1.0) / 2.0  # map [-1,1] -> [0,1]
    # ORB match score [0,1]
    orb_score = orb_match_score(descriptors[i], descriptors[j])
    # Composite: give SSIM strongest weight, then hist, then ORB
    sim = 0.5 * ssim_val + 0.3 * hist_score + 0.2 * orb_score
    # clamp [0,1]
    sim = float(max(0.0, min(1.0, sim)))
    return (i, j, sim)

# -------------------------
# Ordering: bidirectional greedy + 2-opt
# -------------------------
def bidirectional_build(sim_mat):
    n = sim_mat.shape[0]
    unused = set(range(n))
    # anchor: index with minimum average similarity (likely true start)
    avg_sim = sim_mat.mean(axis=1)
    anchor = int(np.argmin(avg_sim))
    seq = deque([anchor])
    unused.remove(anchor)

    # grow alternately on both ends
    while unused:
        # left candidate: best similarity to current left end
        left = seq[0]
        right = seq[-1]

        # find best unmatched for left
        best_left = None
        best_left_score = -1.0
        for u in unused:
            s = sim_mat[u, left]
            if s > best_left_score:
                best_left_score = s
                best_left = u

        # find best unmatched for right
        best_right = None
        best_right_score = -1.0
        for u in unused:
            s = sim_mat[right, u]
            if s > best_right_score:
                best_right_score = s
                best_right = u

        # choose which side to attach (where similarity is higher)
        if best_left_score >= best_right_score:
            seq.appendleft(best_left)
            unused.remove(best_left)
        else:
            seq.append(best_right)
            unused.remove(best_right)

    return list(seq)

def two_opt(order, sim_mat, max_iters=500):
    n = len(order)
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(1, n - 2):
            for j in range(i + 1, n):
                a, b = order[i - 1], order[i]
                c, d = order[j - 1], order[j] if j < n else None
                curr = sim_mat[a, b]
                if d is not None:
                    curr += sim_mat[c, d]
                new = sim_mat[a, c]
                if d is not None:
                    new += sim_mat[b, d]
                if new > curr + 1e-9:
                    order[i:j] = reversed(order[i:j])
                    improved = True
        if not improved:
            break
    return order

# -------------------------
# Main
# -------------------------
def main(args):
    t0 = time.time()
    video_path = Path(args.input)
    out_dir = Path("frames")
    out_dir.mkdir(exist_ok=True)

    print("1) Extracting frames...")
    frames, fps = extract_frames(video_path, out_dir)
    n = len(frames)
    print(f"   extracted {n} frames, fps={fps}")

    # Precompute resized grayscale images, color histograms, ORB descriptors
    print("2) Computing per-frame features...")
    gray_resized = [None] * n
    color_hists = [None] * n
    descriptors = [None] * n
    for i, f in enumerate(tqdm(frames, desc="  frames")):
        img, img_gray = load_image_for_features(f, resize=(256,256))
        gray_resized[i] = img_gray
        color_hists[i] = color_histogram(img).astype('float32')
        descriptors[i] = orb_descriptors(img_gray, nfeatures=args.orb_features)

    # Prepare pair indices for upper triangle
    print("3) Computing pairwise similarities (parallel)...")
    pair_list = [(i,j) for i in range(n) for j in range(i+1, n)]
    worker = partial(pair_similarity, gray_images_resized=gray_resized,
                     color_hists=color_hists, descriptors=descriptors)

    pool_workers = min(cpu_count(), args.workers)
    sim_mat = np.zeros((n, n), dtype=float)
    with Pool(pool_workers) as p:
        for (i,j,sim) in tqdm(p.imap(worker, pair_list), total=len(pair_list), desc="  pairs"):
            sim_mat[i,j] = sim
            sim_mat[j,i] = sim
    np.fill_diagonal(sim_mat, 1.0)

    # 4) Build order bidirectionally
    print("4) Building order (bidirectional)...")
    order = bidirectional_build(sim_mat)

    # 5) Local refinement (2-opt)
    print("5) Refining order with 2-opt...")
    order = two_opt(order, sim_mat, max_iters=args.twoopt_iters)

    # 6) Optional: rotate so that the average similarity to next frame is high at start
    # Compute adjacent sims and pick rotation that maximizes average adjacent similarity
    print("6) Selecting best rotation (to fix residual start issues)...")
    best_avg = -1.0
    best_rot = 0
    adj_sims = [sim_mat[order[i], order[(i+1)%n]] for i in range(n)]
    # try all rotations (cheap for n=300)
    for r in range(n):
        rotated = order[r:] + order[:r]
        avg_adj = np.mean([sim_mat[rotated[i], rotated[i+1]] for i in range(n-1)])
        if avg_adj > best_avg:
            best_avg = avg_adj
            best_rot = r
    order = order[best_rot:] + order[:best_rot]
    print(f"   applied rotation {best_rot} (avg adjacent sim={best_avg:.4f})")

    # Write reconstructed video
    print("7) Writing reconstructed video...")
    first_img = cv2.imread(frames[0])
    h, w = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(args.output, fourcc, args.fps, (w,h))
    for idx in order:
        img = cv2.imread(frames[idx])
        out_vid.write(img)
    out_vid.release()

    total = time.time() - t0
    print(f"Done. Output written to {args.output}")
    print(f"Total runtime: {total:.2f}s")

    # Save run log
    with open("run_log.txt", "w") as f:
        f.write(f"Input: {args.input}\nFrames: {n}\nFPS detected: {fps}\n")
        f.write(f"Workers: {pool_workers}\nORB features: {args.orb_features}\n")
        f.write(f"Two-opt iters: {args.twoopt_iters}\n")
        f.write(f"Total runtime: {total:.2f}s\n")
        f.write(f"Output: {args.output}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="jumbled_video.mp4")
    parser.add_argument("--output", type=str, default="reconstructed_best.mp4")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--orb_features", type=int, default=500)
    parser.add_argument("--twoopt_iters", type=int, default=600)
    args = parser.parse_args()
    main(args)
