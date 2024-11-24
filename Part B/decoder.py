

import numpy as np
from PIL import Image
import math

def decoder(mask_path, masked_image_path, orig_image_path, save_path):
    """
    DECODER function in Python
    Takes a mask and an image with some pixel values missing.
    Generates an estimate of the original image using homogeneous diffusion.
    Computes the PSNR (Peak Signal-to-Noise Ratio) for the restored image.
    
    Parameters:
    mask_path: str, Path to the mask image (must have a .pbm extension).
    masked_image_path: str, Path to the masked image.
    orig_image_path: str, Path to the original image.
    save_path: str, Path to save the restored image.
    """
    # Load images
    res_im = np.array(Image.open(masked_image_path))
    mask_im = np.array(Image.open(mask_path))
    orig_im = np.array(Image.open(orig_image_path))

    # Ensure original image is in RGB (3 channels)
    if orig_im.shape[2] == 4:  # If the image has 4 channels (RGBA)
        orig_im = orig_im[:, :, :3]  # Remove alpha channel

    # Homogeneous Diffusion Parameters
    delta_t = 0.09
    max_time = 500

    # Convert mask to double to avoid calculation loss
    msk = mask_im.astype(float)
    if len(msk.shape) == 2:  # Grayscale mask
        msk = np.stack((msk, msk, msk), axis=-1)  # Convert to 3-channel
    res_im = res_im.astype(float)

    m, n, _ = res_im.shape

    # Perform Homogeneous Diffusion
    for t in np.arange(0, max_time, delta_t):
        # Calculate second derivatives
        res_xx = res_im[:, np.r_[1:n, n-1], :] - 2 * res_im + res_im[:, np.r_[0, 0:n-1], :]
        res_yy = res_im[np.r_[1:m, m-1], :, :] - 2 * res_im + res_im[np.r_[0, 0:m-1], :, :]
        Lap = res_xx + res_yy

        # Effective divergence
        div = delta_t * Lap

        # Update only where mask is True
        res_im += div * msk

    # Convert back to integer values
    res_im = np.clip(res_im, 0, 255).astype(np.uint8)

    # Calculate PSNR
    mse = np.mean((orig_im.astype(float) - res_im.astype(float)) ** 2)
    psnr = 10.0 * math.log10((255.0 ** 2) / mse)
    print(f"PSNR: {psnr:.2f}")

    # Save the restored image
    restored_image = Image.fromarray(res_im)
    restored_image.save(save_path)

    # Display images (optional)
    orig_image = Image.fromarray(orig_im)
    orig_image.show(title='Original Image')
    restored_image.show(title='Restored Image')

# Example usage
decoder('result/m_im1.pbm', 'result/res_im1.jpg', 'data/im1.png', 'result/restored_im1.png')
decoder('result/m_im2.pbm', 'result/res_im2.jpg', 'data/im2.png', 'result/restored_im2.png')
decoder('result/m_im3.pbm', 'result/res_im3.jpg', 'data/im3.png', 'result/restored_im3.png')
decoder('result/m_im4.pbm', 'result/res_im4.jpg', 'data/im4.png', 'result/restored_im4.png')
decoder('result/m_im5.pbm', 'result/res_im5.jpg', 'data/im5.png', 'result/restored_im5.png')

