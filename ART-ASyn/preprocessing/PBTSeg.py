import os, glob
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes

# ----------------------------- helpers ----------------------------- #

def border_touch_fraction(region, label_image):
    convex_image = region.convex_image
    top, left, bottom, right = region.bbox
    convex_mask = np.zeros_like(label_image, dtype=bool)
    convex_mask[top:bottom, left:right] = convex_image
    region = regionprops(convex_mask.astype(np.uint8))[0]
    coords = region.coords
    w, h = label_image.shape
    top = coords[:, 0] == 0
    bottom = coords[:, 0] == w - 1
    left = coords[:, 1] == 0
    right = coords[:, 1] == h - 1
    border_pixels = top | bottom | left | right
    fraction = np.sum(border_pixels) / max(1, region.perimeter)
    return fraction

def good_lung_shape(region, label_image):
    if region.area <= 2000 or region.area >= 10000:
        return False
    height, width = region.image.shape
    if height <= width:
        return False
    if region.solidity <= 0.6:
        return False
    w, h = label_image.shape
    y, x = region.centroid
    w_1_6, h_1_6 = int(w / 6), int(h / 6)
    w_5_6, h_5_6 = w - w_1_6, h - h_1_6
    if not w_1_6 < y < w_5_6 or not h_1_6 < x < h_5_6:
        return False
    btf = border_touch_fraction(region, label_image)
    if btf >= 0.15:
        return False
    return True

def lung_segmentation(img):
    right_lung, left_lung = np.zeros_like(img), np.zeros_like(img)
    right_lung_area, left_lung_area = 0, 0
    thres = 50
    while True:
        if thres > 160:
            break
        mask = (img <= thres)
        label_image = label(mask)
        regions = regionprops(label_image)
        regions = list(filter(lambda x: good_lung_shape(x, label_image), regions))
        regions = sorted(regions, key=lambda x: x.area, reverse=True)
        for region in regions[:2]:
            top, left, bottom, right = region.bbox
            if left < (256 // 3) and region.area > left_lung_area:
                right_lung = label_image == region.label
                left_lung_area = region.area
            elif right > (256 // 3 * 2) and region.area > right_lung_area:
                left_lung = label_image == region.label
                right_lung_area = region.area
        if len(regions) == 0 and thres > 130:
            break
        thres += 5
    right_lung = binary_fill_holes(right_lung)
    left_lung = binary_fill_holes(left_lung)
    return right_lung, left_lung

# ----------------------------- per-image ----------------------------- #

def process_one_image(img_path: str, img_size: int, out_subdir: str) -> int:
    try:
        img_path = Path(img_path)
        suffix = ".npy"
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
        right_lung, left_lung = lung_segmentation(img)
        lung_label = np.zeros_like(img, dtype=np.uint8)
        lung_label[right_lung] = 1
        lung_label[left_lung] = 2
        save_dir = img_path.parent.parent / out_subdir
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{img_path.stem}{suffix}"
        np.save(save_path, lung_label)
        return 1
    except Exception:
        return 0

# ----------------------------- dataset wrapper ----------------------------- #

def process_dataset_lung_masks(
    root_dir: str,
    dataset: str,
    split: str = "train",
    subset: str = "healthy",
    out_subdir: str = "lung_mask",
    img_size: int = 256,
    max_images: Optional[int] = None,
    n_workers: int = 8,
) -> int:
    root = Path(root_dir).expanduser().resolve()
    img_glob = str(root / dataset / split / subset / "*")
    img_paths = glob.glob(img_glob)
    if max_images is not None:
        img_paths = img_paths[:max_images]
    if len(img_paths) == 0:
        print(f"[WARN] No images found at: {img_glob}")
        return 0
    processed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(process_one_image, p, img_size, out_subdir) for p in img_paths]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Lung segmentation"):
            processed += f.result() or 0
    print(f"[DONE] Processed {processed} studies. Output under '{out_subdir}/'.")
    return processed

# ----------------------------- CLI ----------------------------- #

if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser()
    cli.add_argument("--root_dir", default="../../data", type=str)
    cli.add_argument("--dataset", default="CheXpert", type=str)
    cli.add_argument("--split", default="train", type=str)
    cli.add_argument("--subset", default="healthy", type=str)
    cli.add_argument("--out_subdir", default="healthy_lung_mask", type=str)
    cli.add_argument("--img_size", default=256, type=int)
    cli.add_argument("--max_images", default=None, type=int)
    cli.add_argument("--n_workers", default=8, type=int)
    args = cli.parse_args()

    process_dataset_lung_masks(
        root_dir=args.root_dir,
        dataset=args.dataset,
        split=args.split,
        subset=args.subset,
        out_subdir=args.out_subdir,
        img_size=args.img_size,
        max_images=args.max_images,
        n_workers=args.n_workers,
    )
