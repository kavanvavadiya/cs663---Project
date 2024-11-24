import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

def encoder(image_path, mask_path, masked_image_path):
    """
    Takes an input image (3 channels) and generates a mask.
    Saves the mask and the masked image.
    mask_path must have a .pbm extension.
    """
    # Read the input image
    im = cv2.imread(image_path)
    m, n, _ = im.shape

    # Get the edges using Canny edge detection
    gray_im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    ed_im = cv2.Canny(gray_im, 100, 200)  # Edge detection

    # Initialize the mask with ones
    mask_im = np.ones((m, n, 3), dtype=np.uint8)

    # Exclude neighbors of pixels around edges in the mask
    window = 3
    pd = (window - 1) // 2

    # Pad the mask
    pmsk = np.pad(mask_im, ((pd, pd), (pd, pd), (0, 0)), mode='constant', constant_values=1)

    for i in range(pd, pd + m):
        for j in range(pd, pd + n):
            # Exclude the neighbor if edge is found
            if ed_im[i - pd, j - pd]:
                pmsk[i - pd:i + pd + 1, j - pd:j + pd + 1, :] = 0

    # Extract the mask out of the padded array
    mask_im = pmsk[pd:pd + m, pd:pd + n, :]

    # Exclude the boundary pixels of the image in the mask
    mask_im[0, :, :] = 0
    mask_im[-1, :, :] = 0
    mask_im[:, 0, :] = 0
    mask_im[:, -1, :] = 0

    # Generate the result image with the negation of the mask applied
    res_im = (1 - mask_im) * im

    # Display the mask and the result
    plt.figure()
    plt.imshow(cv2.cvtColor(mask_im * 255, cv2.COLOR_BGR2RGB))
    plt.title('Mask')
    plt.show()

    plt.figure()
    plt.imshow(cv2.cvtColor(res_im, cv2.COLOR_BGR2RGB))
    plt.title('Result')
    plt.show()


    cv2.imwrite(masked_image_path, res_im)
    mask_pbm = (mask_im[:, :, 0] * 255).astype(np.uint8)  # Convert to 2D binary image
    mask_image = Image.fromarray(mask_pbm).convert("1")  # Convert to 1-bit mode for PBM
    mask_image.save(mask_path)


# Example usage:
encoder('data/im1.png', 'result/m_im1.pbm', 'result/res_im1.jpg')
encoder('data/im2.png', 'result/m_im2.pbm', 'result/res_im2.jpg')
encoder('data/im3.png', 'result/m_im3.pbm', 'result/res_im3.jpg')
encoder('data/im4.png', 'result/m_im4.pbm', 'result/res_im4.jpg')
encoder('data/im5.png', 'result/m_im5.pbm', 'result/res_im5.jpg')
