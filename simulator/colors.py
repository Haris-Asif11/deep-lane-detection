"""
Coloring functions used to create a 3-channel numpy array representing an
RGB image from a layer represented by a bitmask. The output numpy array
supports 8-bit color depth (uint8).

"""
import numpy as np
import random
from skimage.io import imread
from skimage.transform import resize
from utils.file_handling import get_image_list

def _clip_color_tuple(color):
    """Return a (3,) uint8 array, clipped to [0, 255]. Accepts any ints/floats."""
    c = np.array(color, dtype=np.int16)   # allow negatives/ >255 temporarily
    c = np.clip(c, 0, 255).astype(np.uint8)
    if c.shape != (3,):
        raise ValueError(f"color must be a 3-tuple (r,g,b), got shape {c.shape}")
    return c


def color_w_constant_color(fmask, color):
    """
    Color whole layer with a constant color.

    Input:
    fmask -- binary mask (2D) indicating the shape to color (1 where colored)
    color -- (r,g,b) tuple (any ints/floats; will be clipped to [0,255])

    Output:
    uint8 numpy array of shape (x,y,3)
    """
    x, y = fmask.shape
    img = np.zeros((x, y, 3), dtype=np.uint8)

    # Ensure safe values and types
    color_u8 = _clip_color_tuple(color)

    # Boolean mask of where to apply color
    mask = fmask.astype(bool)

    # Assign per-pixel color on masked positions (N,3) <- (3,)
    # This avoids unsafe arithmetic on uint8.
    img[mask] = color_u8
    return img


def color_w_noisy_color(fmask, mean, deviation):
    """
    Colors layer with 'mean' color, then adds uniform integer noise in
    [-deviation, +deviation]. Output is clipped to [0,255], dtype uint8.

    Input:
    fmask -- binary mask (2D)
    mean  -- (r,g,b) tuple (will be clipped to [0,255])
    deviation -- non-negative int
    """
    base = color_w_constant_color(fmask, mean).astype(np.int16)  # promote to avoid overflow

    # Generate noise; if you want the same noise per channel, create 2D and expand
    x, y = fmask.shape
    noise2d = np.random.randint(-int(deviation), int(deviation) + 1, size=(x, y), dtype=np.int16)
    noise = np.repeat(noise2d[:, :, None], 3, axis=2)  # (x,y,3), same noise across channels

    out = base + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _ensure_rgb_uint8(img):
    """
    Ensure image is 3-channel RGB uint8.
    - If grayscale, stack to 3 channels.
    - If has alpha, drop alpha.
    - If float, scale if needed and cast to uint8.
    """
    arr = np.asarray(img)

    # If float-like in [0,1], scale to [0,255]. If already [0,255], just clip.
    if np.issubdtype(arr.dtype, np.floating):
        # Try to infer scale
        if arr.max() <= 1.0:
            arr = (arr * 255.0)
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    elif arr.dtype != np.uint8:
        # Generic cast with clipping
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Handle channels
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)  # grayscale -> RGB
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]  # drop alpha
        elif arr.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel count: {arr.shape[2]}")

    return arr


def color_w_image(fmask, folder, rotate):
    """
    Fill masked pixels with pixels from a random image in 'folder'.

    Input:
    fmask  -- binary mask (2D)
    folder -- folder path containing candidate images
    rotate -- if 1, rotate the selected image by k*90 degrees, with k in {0,1,2,3}

    Output:
    uint8 numpy array of shape (x,y,3)
    """
    x, y = fmask.shape
    img = np.zeros((x, y, 3), dtype=np.uint8)

    image_files = get_image_list(folder)
    if not image_files:
        # No images found; just return black image
        return img

    true_image_path = random.choice(image_files)
    true_image = imread(true_image_path)

    # Ensure RGB uint8 and resize to mask size (keep values, no normalization)
    true_image = _ensure_rgb_uint8(true_image)
    if (true_image.shape[0] != x) or (true_image.shape[1] != y):
        true_image = resize(
            true_image,
            (x, y),
            preserve_range=True,
            anti_aliasing=True
        )
        true_image = _ensure_rgb_uint8(true_image)

    if rotate == 1:
        # k in {0,1,2,3}
        k = random.randint(0, 3)
        if k:
            true_image = np.rot90(true_image, k)

    # Apply mask
    mask = fmask.astype(bool)
    img[mask] = true_image[mask]

    return img


# Public API
COLOR_FCT_REGISTRY = {
    'constant': color_w_constant_color,
    'noisy'   : color_w_noisy_color,
    'image'   : color_w_image
}