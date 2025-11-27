import cv2
import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from skimage.segmentation import find_boundaries

def brighten_border(img):
    # Brighten outside border
    h, w = img.shape
    
    gradient = np.ones_like(img, dtype=np.float32) * 1.6
    h_1_4, w_1_5 = int(h / 4), int(w / 5)
    x_start, y_start = h_1_4, w_1_5//2
    x_end, y_end = h - h_1_4//2, w - w_1_5//2
    gradient[x_start:x_end, y_start:y_end] = 1
    
    kernel_size = h//4 * 2 + 1
    gradient = cv2.GaussianBlur(gradient, (kernel_size, kernel_size), sigmaX=0)
    bright_img = (img.astype(np.float32) * gradient).clip(0, 255).astype(np.uint8)

    return bright_img

def border_touch_fraction(region, label_image):
    """
    Returns the fraction of region pixels that lie on the image border.
    """
    # Convert region to convex region
    convex_image = region.convex_image
    top, left, bottom, right = region.bbox
    convex_mask = np.zeros_like(label_image, dtype=bool)
    convex_mask[top:bottom, left:right] = convex_image
    
    region = regionprops(convex_mask.astype(np.uint8))[0]
    
    
    # Find pixels of convex regions which is on the border
    coords = region.coords
    w, h = label_image.shape

    top = coords[:, 0] == 0
    bottom = coords[:, 0] == w - 1
    left = coords[:, 1] == 0
    right = coords[:, 1] == h - 1

    border_pixels = top | bottom | left | right
    
    # Fraction perimeter which is on the border
    fraction = np.sum(border_pixels) / max(1, region.perimeter)
    return fraction


def good_lung_shape(region, label_image):
    """
    Good lung shape
       - 1. Area between 2000 and 10000
       - 2. Height > width
       - 3. Not very convave -> solidity > 0.7 (solidity = area / convex area)
       - 4. Centorid away from border -> centroid is within 2/3 area of image (away from 1/6 borders)
       - 5. Less than 20% perimeter connected to border (not very attach to border)
    """
    
    from utils import display_images
    
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

    if border_touch_fraction(region, label_image) >= 0.15:
        return False
    
    return True

def lung_segmentation(img):
    h, w = img.shape
    w_1_3, w_2_3 = w // 3, w - w // 3

    right_lung, left_lung = np.zeros_like(img), np.zeros_like(img)
    right_lung_area, left_lung_area = 0, 0

    thres = 50
    while True:
        if thres > 160:
            break

        mask = (img <= thres)
        label_image = label(mask)

        # Get independent parts of image
        regions = regionprops(label_image)

        # Filter regions which are candidates as lung segmentations
        regions = list(filter(lambda x: good_lung_shape(x, label_image), regions))
        regions = sorted(regions, key=lambda x: x.area, reverse=True)

        # Process regions with top-2 areas
        # Classify as left/right lung by bounding box positions AND update their regions
        for region in regions[:2]:            
            top, left, bottom, right = region.bbox
            if left < w_1_3 and region.area > left_lung_area: # left side -> right lung
                right_lung = label_image==region.label
                left_lung_area = region.area
            elif right > w_2_3 and region.area > right_lung_area: # right side -> left lung
                left_lung = label_image==region.label
                right_lung_area = region.area
        
        # Break early if no updates is avaliable
        if len(regions) == 0 and thres > 130:
            break
        thres += 5
    
    right_lung = binary_fill_holes(right_lung)
    left_lung = binary_fill_holes(left_lung)

    return right_lung, left_lung