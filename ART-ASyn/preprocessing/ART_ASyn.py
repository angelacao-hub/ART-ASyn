"""
Process a chest X-ray dataset to synthesize anomaly images.
"""

import os, glob, argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageEnhance
from scipy.spatial import cKDTree
from concurrent.futures import ProcessPoolExecutor, as_completed

# project-specific utils (assumed available)
from cxr_utils import Brush, CXRImage  # Brush must accept (filepath, color=list[int])

# ----------------------------- helpers & effects ----------------------------- #

def _seed_worker(seed: Optional[int], worker_idx: int):
    # Keep BLAS/OpenCV from oversubscribing
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    # Set numpy seed for reproducibility per worker
    if seed is not None:
        np.random.seed(seed + worker_idx)

def normalize(array: np.ndarray) -> np.ndarray:
    if not array.any():
        return array
    amin = array.min()
    amax = array.max()
    if amax == amin:
        return np.zeros_like(array, dtype=np.float32)
    return (array - amin) / (amax - amin)

def random_curve(p0: np.ndarray, p1: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    p0, p1: np.ndarray shape (2,) as (x,y)
    Returns:
        curve: [N,2] int points along a quadratic Bezier
        approx_len: rough curve length for spacing decisions
    """
    p0 = np.asarray(p0, dtype=np.float32)
    p1 = np.asarray(p1, dtype=np.float32)
    x_displace = np.random.uniform(low=-30, high=30)
    y_displace = np.random.uniform(low=-30, high=30)
    p_mid = (p0 + p1) / 2 + np.array([x_displace, y_displace], dtype=np.float32)

    chord_len = float(np.linalg.norm(p1 - p0))
    control_len = float(np.linalg.norm(p_mid - (p0 + p1) / 2))
    approx_len = chord_len + control_len  # overestimate is fine

    t = np.linspace(0, 1, 100, dtype=np.float32)[:, None]
    curve = ((1 - t) ** 2) * p0 + 2 * (1 - t) * t * p_mid + (t ** 2) * p1
    curve = np.round(curve).astype(np.int32)
    return curve, approx_len

# global collector for debug dots
all_points: List[Tuple[int, int]] = []

def compose_anomaly(p0, p1, img_np: np.ndarray, base_brush: Brush, opacity: float = 1.0) -> Image.Image:
    """
    Build a layered, brush-stamped anomaly along a random curve between p0 and p1.
    """
    H, W = img_np.shape[:2]
    base_brush_1 = base_brush.resize(width=80)
    base_brush_2 = base_brush.resize(width=50)

    curve, curve_len = random_curve(p0, p1)

    spacing_1 = max(1, int(base_brush_1.width * 0.05))
    spacing_2 = max(1, int(base_brush_2.width * 0.05))
    # sample along curve proportional to curve length / spacing
    idx_pool = np.arange(len(curve))
    n1 = min(len(curve), max(1, int(curve_len / spacing_1)))
    n2 = min(len(curve), max(1, int(curve_len / spacing_2)))
    points_1 = curve[np.random.choice(idx_pool, size=n1, replace=False)]
    points_2 = curve[np.random.choice(idx_pool, size=n2, replace=False)]

    anomaly = Image.new("RGBA", (W, H), (0, 0, 0, 0))  # blank canvas RGBA

    global all_points
    all_points.extend([(int(x), int(y)) for x, y in points_1])

    for x, y in points_1:
        brush = Brush.brush_jitter(base_brush_1)
        top_left = (int(x - brush.width / 2), int(y - brush.height / 2))
        anomaly.alpha_composite(brush.image, dest=top_left)

    for x, y in points_2:
        brush = Brush.brush_jitter(base_brush_2)
        top_left = (int(x - brush.width / 2), int(y - brush.height / 2))
        anomaly.alpha_composite(brush.image, dest=top_left)

    # scale alpha by opacity
    anomaly.putalpha(anomaly.getchannel("A").point(lambda p: int(p * float(opacity))))
    return anomaly

def draw_points(canvas_size: Tuple[int, int], points: List[Tuple[int, int]], dot_radius: int = 5, opacity: float = 0.2) -> Image.Image:
    W, H = canvas_size
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))

    dot_size = dot_radius * 2
    dot = Image.new("RGBA", (dot_size, dot_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(dot)
    draw.ellipse((0, 0, dot_size - 1, dot_size - 1), fill=(255, 255, 255, int(255 * opacity)))

    for x, y in points:
        x = int(x + np.random.randint(low=-10, high=10))
        y = int(y + np.random.randint(low=-10, high=10))
        top_left = (int(x - dot_radius), int(y - dot_radius))
        canvas.alpha_composite(dot, dest=top_left)

    # reset collector
    global all_points
    all_points = []
    return canvas

def voronoi_crystallize_pil(pil_img: Image.Image, num_seeds: int = 500) -> Image.Image:
    image = np.array(pil_img)
    H, W = image.shape[:2]

    coords = np.mgrid[0:H, 0:W].reshape(2, -1).T
    seeds_yx = np.random.randint(0, [H, W], size=(num_seeds, 2))
    tree = cKDTree(seeds_yx)
    _, labels = tree.query(coords, k=1)
    labels = labels.reshape(H, W)

    output = np.zeros_like(image)
    for i in range(num_seeds):
        mask = labels == i
        if np.any(mask):
            mean_color = image[mask].mean(axis=0).astype(np.uint8)
            output[mask] = mean_color

    return Image.fromarray(output)

def random_anomaly(cxr_image: CXRImage, img_np: np.ndarray, lung_label: np.ndarray, return_intermediate: bool = False):
    """
    Compose a stylized anomaly constrained to lung regions with rib-intensity modulation.
    """
    H, W = img_np.shape[:2]
    brush_color = [int(cxr_image.percentile(85))] * 3
    base_brush = Brush(filepath="texture.jpg", color=brush_color)

    # sample segment pairs inside lungs
    while True:
        n_left = np.random.choice([*range(5), *range(10, 15)], p=[0.2] + [0.8 / 9] * 9)
        n_right = np.random.choice([*range(5), *range(10, 15)], p=[0.2] + [0.8 / 9] * 9)

        left_points = cxr_image.left_lung.get_random_point(n=n_left * 2).reshape((n_left, 2, 2)) if cxr_image.left_lung.mask.any() else np.empty((0, 2, 2))
        right_points = cxr_image.right_lung.get_random_point(n=n_right * 2).reshape((n_right, 2, 2)) if cxr_image.right_lung.mask.any() else np.empty((0, 2, 2))
        points = np.vstack((left_points, right_points))
        if len(points) > 0:
            break

    anomaly = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    for p0, p1 in points:
        opacity = 0.95
        anomaly.alpha_composite(compose_anomaly(p0, p1, img_np, base_brush, opacity=opacity))

    intermediates = []
    if return_intermediate:
        intermediates = [draw_points((W, H), all_points)]

    # effects: blur + crystallize layers
    blurred = anomaly.filter(ImageFilter.GaussianBlur(radius=8))
    crystalize = voronoi_crystallize_pil(anomaly)
    crystalize_blurred = crystalize.filter(ImageFilter.GaussianBlur(radius=3))

    if return_intermediate:
        intermediates.extend([anomaly.getchannel("A"), blurred.getchannel("A"), crystalize.getchannel("A"), crystalize_blurred.getchannel("A")])

    def scale_alpha(img_pil: Image.Image, factor: float) -> Image.Image:
        return img_pil.getchannel("A").point(lambda p: int(p * factor))

    # reassign scaled alphas
    anomaly.putalpha(scale_alpha(anomaly, 0.4))
    crystalize.putalpha(scale_alpha(crystalize, 0.5))
    crystalize_blurred.putalpha(scale_alpha(crystalize_blurred, 0.5))

    # composite layers
    anomaly.alpha_composite(blurred)
    anomaly.alpha_composite(crystalize)
    anomaly.alpha_composite(crystalize_blurred)
    anomaly.alpha_composite(crystalize)
    anomaly.alpha_composite(blurred)

    # mask dilation + blur for soft edges
    expanded_mask = cv2.dilate((lung_label > 0).astype(np.uint8), np.ones((7, 7), np.uint8))
    blurred_mask = cv2.GaussianBlur(expanded_mask.astype(np.float32), (31, 31), sigmaX=0)

    # rib brightness modulation
    rib_intensity = np.clip(img_np.astype(np.float32) / 255.0, 0.0, 1.0)
    rib_contrast = 0.5 * rib_intensity + 0.5

    if return_intermediate:
        intermediates.append(Image.fromarray((rib_intensity * blurred_mask * 255).astype(np.uint8)))

    # update alpha by lung + rib mask
    alpha_np = np.array(anomaly.getchannel("A"), dtype=np.float32)
    new_alpha = alpha_np * rib_contrast * blurred_mask
    new_alpha = np.clip(new_alpha, 0, 255).astype(np.uint8)
    new_alpha_img = Image.fromarray(new_alpha)
    new_alpha_img = ImageEnhance.Brightness(new_alpha_img).enhance(1.5)
    anomaly.putalpha(new_alpha_img)

    anomaly_image = cxr_image.image.convert("RGBA")
    anomaly_image.alpha_composite(anomaly)
    anomaly_image = ImageChops.lighter(anomaly_image, Image.fromarray(img_np).convert("RGBA"))

    if return_intermediate:
        return anomaly, anomaly_image, intermediates
    return anomaly, anomaly_image

# ----------------------------- main processing ------------------------------ #

def process_one_image(
    img_path: str,
    img_size: int,
    subset: str,
    mask_subdir: str,
    out_subdir: str,
) -> int:
    """
    Returns 1 if processed, 0 if skipped/failed.
    """
    try:
        img_path = Path(img_path)
        suffix = img_path.suffix or ".png"

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # Find paired mask
        label_path = Path(str(img_path).replace(f"/{subset}/", f"/{mask_subdir}/")).with_suffix(".npy")
        if not label_path.exists():
            # fallback
            label_path = img_path.parent.parent / mask_subdir / (img_path.stem + ".npy")
        if not label_path.exists():
            return 0

        lung_label = np.load(label_path)
        if lung_label.shape != img.shape:
            lung_label = cv2.resize(lung_label.astype(np.uint8), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        if not lung_label.any():
            return 0

        cxr_image = CXRImage(img, lung_label=lung_label)

        study_id = label_path.stem
        save_dir = label_path.parent.parent / out_subdir / study_id
        save_dir.mkdir(parents=True, exist_ok=True)

        n_anomaly = int(np.random.choice([1, 2, 3]))
        for i in range(n_anomaly):
            anomaly, anomaly_image = random_anomaly(cxr_image, img, lung_label)

            alpha = np.array(anomaly)[:, :, -1]
            mask_bin = (alpha > 80).astype(np.uint8) * 255

            Image.fromarray(mask_bin).save(save_dir / f"mask_{i}{suffix}")
            anomaly_image.convert("RGB").save(save_dir / f"anomaly_{i}{suffix}")

        return 1
    except Exception:
        # Optional: log the path or traceback
        return 0

def process_dataset_parallel(
    root_dir: str,
    dataset: str,
    split: str = "train",
    subset: str = "healthy",
    mask_subdir: str = "healthy_lung_mask",
    out_subdir: str = "synthetic_anomaly",
    img_size: int = 256,
    max_images: Optional[int] = None,
    seed: Optional[int] = None,
    n_workers: int = 8,
) -> int:
    """
    Parallel version of process_dataset. Returns number of studies processed.
    """
    root = Path(root_dir).expanduser().resolve()
    img_glob = str(root / dataset / split / subset / "*")
    img_paths = glob.glob(img_glob)
    if max_images is not None:
        img_paths = img_paths[:max_images]

    if len(img_paths) == 0:
        print(f"[WARN] No images found at: {img_glob}")
        return 0

    processed = 0
    # Important for Windows/macOS. You can also set this in your __main__ guard.
    try:
        import multiprocessing as mp
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_seed_worker, initargs=(seed, 0)) as ex:
        # Submit with per-task worker index baked into args by enumerating
        futures = []
        for idx, p in enumerate(img_paths):
            # We pass seed only via initializer; if you need per-task seeding,
            # you can encode it into the path or add to kwargs.
            futures.append(
                ex.submit(
                    process_one_image,
                    p, img_size, subset, mask_subdir, out_subdir
                )
            )

        for f in tqdm(as_completed(futures), total=len(futures), desc="Parallel processing"):
            processed += f.result() or 0

    print(f"[DONE] (parallel) Processed {processed} studies. Output under '{out_subdir}/<study_id>/*'.")
    return processed

# Example CLI usage:
if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser()
    cli.add_argument("--root_dir", default="../../data", type=str)
    cli.add_argument("--dataset", default="CheXpert", type=str)
    cli.add_argument("--split", default="train", type=str)
    cli.add_argument("--subset", default="healthy", type=str)
    cli.add_argument("--mask_subdir", default="healthy_lung_mask", type=str)
    cli.add_argument("--out_subdir", default="synthetic_anomaly", type=str)
    cli.add_argument("--img_size", default=256, type=int)
    cli.add_argument("--max_images", default=None, type=int)
    cli.add_argument("--seed", default=None, type=int)
    cli.add_argument("--n_workers", default=8, type=int)
    args = cli.parse_args()

    process_dataset_parallel(
        root_dir=args.root_dir,
        dataset=args.dataset,
        split=args.split,
        subset=args.subset,
        mask_subdir=args.mask_subdir,
        out_subdir=args.out_subdir,
        img_size=args.img_size,
        max_images=args.max_images,
        seed=args.seed,
        n_workers=args.n_workers,
    )