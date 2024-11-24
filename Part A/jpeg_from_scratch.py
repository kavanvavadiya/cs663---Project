import heapq
from collections import defaultdict
import numpy as np
import cv2  # OpenCV library for image processing
from scipy.fftpack import dct, idct
import os
import matplotlib.pyplot as plt
import re
import sys
import json



def preprocess_grayscale_image(image_path):
    """
    Preprocess a grayscale image for JPEG compression, ensuring dimensions are multiples of 8 by clipping.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Could not load image. Check the file path.")

    # Ensure dimensions are multiples of 8 by clipping
    height, width = image.shape
    height_clipped = (height // 8) * 8
    width_clipped = (width // 8) * 8
    image = image[:height_clipped, :width_clipped]
    
    # Center pixel values to [-128, 127]
    centered_image = np.array(image,dtype=np.float32)-128

    return centered_image

def perform_dct(image, block_size=8):
    """
    Perform the Discrete Cosine Transform (DCT) on an image.
    """
    height, width = image.shape
    blocks = [
        dct(dct(image[i:i+block_size, j:j+block_size].T, norm='ortho').T, norm='ortho')
        for i in range(0, height, block_size)
        for j in range(0, width, block_size)
    ]
    scaling_factor = 1 #/ (block_size * block_size)
    scaled_dct_blocks = [block * scaling_factor for block in blocks]
    return scaled_dct_blocks

def quantize_dct_blocks(dct_blocks, Q, block_size=8):
    """
    Quantize the DCT coefficients using a quality-adjusted quantization matrix.
    
    Args:
        dct_blocks (list): List of 2D numpy arrays containing DCT coefficients.
        Q (int): Quality factor (1 to 100).
        block_size (int): Block size (default is 8x8 for JPEG).
        
    Returns:
        list: List of quantized DCT coefficients in zigzag order for each block.
    """
    # Base quantization matrix for luminance (standard JPEG matrix)
    Q_base = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ], dtype=np.float32)
    
    # Adjust the quantization matrix based on the quality factor Q
    if Q < 100:
        scaling_factor = 50 / Q
    else:
        scaling_factor = (100 - Q) / 50
    
    # Q_matrix = np.clip(Q_base * scaling_factor+0.5, 1, None)  # Avoid zero values
    Q_matrix = Q_base * scaling_factor+0.5
    Q_matrix[Q_matrix == 0] = 1; # // Prevent divide by 0 error

    
    # Zigzag order mapping (for an 8x8 block)
    zigzag_indices = [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63]
    ]
    zigzag_indices = np.array(zigzag_indices).flatten()

    # Quantize each block and store in zigzag order
    quantized_blocks = []
    for block in dct_blocks:
        # Ensure DCT coefficients are in range [-1024, 1024]
        if np.any(np.abs(block) > 1024):
            raise ValueError("DCT coefficients out of expected range [-1024, 1024].")

        # Quantize the block
        quantized_block = np.round(block / Q_matrix).astype(int)
        
        # Rearrange into zigzag order
        zigzag_block = quantized_block.flatten()[zigzag_indices]
        quantized_blocks.append(zigzag_block)
    
    return quantized_blocks, Q_matrix

def run_length_encode(ac_coefficients):
    """
    Perform Run-Length Encoding (RLE) on the AC coefficients of a block.

    Args:
        ac_coefficients (list): Zigzag-ordered AC coefficients for a single block.

    Returns:
        list: RLE pairs (run-length, value), with EOB and ZRL markers where applicable.
    """
    rle = []
    zero_count = 0

    for coeff in ac_coefficients:
        if coeff == 0:
            zero_count += 1
            # if zero_count == 16:  # Insert ZRL marker for 16 zeros
            #     rle.append((15, 0))
            #     zero_count = 0
        else:
            rle.append((zero_count, coeff))
            # print((zero_count, coeff))
            zero_count = 0

    # Append EOB if there are trailing zeros
    if zero_count > 0:
        rle.append((0, 0))  # EOB marker
    
    return rle

class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq
    
def huffman_tree(freq_dict):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq)
        merged.left, merged.right = node1, node2
        heapq.heappush(heap, merged)
    
    return build_huffman_codes(heap[0])

def build_huffman_codes(tree, prefix="", codes=None):
    if codes is None:
        codes = {}
    if tree.symbol is not None:
        codes[tree.symbol] = prefix
    else:
        if tree.left:
            build_huffman_codes(tree.left, prefix + "0", codes)
        if tree.right:
            build_huffman_codes(tree.right, prefix + "1", codes)
    return codes

def huffman_encode_dc(data, huffman_table):
    """
    Encode data using a Huffman table.

    Args:
        data (list): List of symbols to encode.
        huffman_table (dict): Huffman table mapping symbols to binary codes.

    Returns:
        str: Huffman-encoded binary string.
    """
    return ''.join(huffman_table[symbol] for symbol in data)

def huffman_encode_ac(data, huffman_table):
    """
    Encode data using a Huffman table.

    Args:
        data (list): List of symbols to encode.
        huffman_table (dict): Huffman table mapping symbols to binary codes.

    Returns:
        str: Huffman-encoded binary string.
    """
    return ''.join([str((symbol[0],len(huffman_table[symbol[1]]),huffman_table[symbol[1]])) for symbol in data])

def prepare_bitstream(dct_coefficients, quant_matrix):
    """
    Prepare the JPEG bitstream by applying RLE, Huffman encoding, and metadata.

    Args:
        dct_coefficients (list): List of quantized DCT coefficients (block by block).
        quant_matrix (np.ndarray): Quantization matrix used.

    Returns:
        tuple: (header, bitstream)
    """
    header = {"quantization_matrix": quant_matrix.tolist(), "huffman_tables": {}}
    dc_values = []
    rle_ac_coefficients = []
    
    # Separate DC and AC coefficients
    for i, block in enumerate(dct_coefficients):
        dc_values.append(int(block[0]))  # First coefficient is the DC
        ac_coefficients = block[1:]  # Remaining coefficients are AC
        rle_ac_coefficients.extend(run_length_encode(ac_coefficients))
    
    # Huffman Encoding: Compute frequency tables
    dc_differences = [dc_values[i] - dc_values[i - 1] if i > 0 else dc_values[i] for i in range(len(dc_values))]
    dc_frequency = defaultdict(int)
    ac_frequency = defaultdict(int)
    
    for dc_diff in dc_differences:
        dc_frequency[dc_diff] += 1
    for rle_pair in rle_ac_coefficients:
        ac_frequency[rle_pair[1]] += 1
    
    # Build Huffman tables
    dc_huffman_table = huffman_tree(dc_frequency)
    ac_huffman_table = huffman_tree(ac_frequency)      

    header["huffman_tables"]["dc"] = dc_huffman_table
    header["huffman_tables"]["ac"] = ac_huffman_table
    
    # Encode the data using the Huffman tables
    dc_encoded = huffman_encode_dc(dc_differences, dc_huffman_table)
    ac_encoded = huffman_encode_ac(rle_ac_coefficients, ac_huffman_table)

    # Combine DC and AC streams
    bitstream = dc_encoded + ac_encoded
    
    return header, bitstream

def regenerate_tree(codes):
    head = HuffmanNode()
    
    for symbol, code in codes.items():
        parent = head
        while len(code)>0:
            if code[0] == '0':
                if parent.left is None:
                    left = HuffmanNode()
                    parent.left = left
                parent = parent.left
            else:
                if parent.right is None:
                    right = HuffmanNode()
                    parent.right = right
                parent = parent.right
            code = code[1:]
        parent.symbol = symbol
    
    return head

def huffman_decode_dc(encoded_data, tree):
    decoded_data = []
    node = tree
    for bit in encoded_data:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            decoded_data.append(node.symbol)
            node = tree
    return decoded_data

def huffman_decode_ac(encoded_data, tree):
    decoded_data = []
    ac_coeffs = 0
    for zeros,code_len,code in encoded_data:
        for i in range(zeros):
            decoded_data.append(0)
        node = tree
        for i in range(code_len):
            node = node.left if code[i] == '0' else node.right
        if node.symbol != 0:
            decoded_data.append(node.symbol)
            ac_coeffs += 1 + zeros
        else:
            for i in range(63-ac_coeffs):
                decoded_data.append(0)
            ac_coeffs = 0
    return decoded_data

def decode_and_reconstruct(header, bitstream):
    """
    Decode the JPEG bitstream and reconstruct the image.
    
    Args:
        header (dict): Contains quantization matrix, Huffman tables, and image dimensions.
        bitstream (str): Compressed bitstream to decode.
    
    Returns:
        np.ndarray: Reconstructed grayscale image.
    """
    import numpy as np
    from scipy.fftpack import idct

    # Zigzag reordering table for an 8x8 block
    zigzag_order = [
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63
    ]

    inverse_zigzag_order = np.argsort(zigzag_order)
    
    # Step 1: Extract quantization matrix and Huffman tables
    Q_matrix = np.array(header["quantization_matrix"])
    dc_huffman_table = header["huffman_tables"]["dc"]
    ac_huffman_table = header["huffman_tables"]["ac"]

    # Step 2: Extract image dimensions and block size
    height, width = header["image_dimensions"]
    block_size = 8
    num_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)

    # Splitting bitstream
    idx = bitstream.find('(')
    dc_bitstream = bitstream[:idx]
    ac_bitstream = bitstream[idx:]
    
    dc_huffman_tree = regenerate_tree(dc_huffman_table)
    ac_huffman_tree = regenerate_tree(ac_huffman_table)
    
    # Step 3: Extracting list of tuples from ac_bitstream
    matches = re.findall(r"\((\d+), (\d+), '(.*?)'\)", ac_bitstream)
    ac_tuples = [(int(x[0]), int(x[1]), x[2]) for x in matches]

    # Step 4: Decode the DC coefficients
    dc_differences = huffman_decode_dc(dc_bitstream, dc_huffman_tree)
    ac_coefficients = huffman_decode_ac(ac_tuples,ac_huffman_tree)
    
    dc_coefficients = [dc_differences[0]]
    for diff in dc_differences[1:]:
        dc_coefficients.append(dc_coefficients[-1] + diff)
    print("done")

    if len(dc_coefficients)!= num_blocks and len(ac_coefficients) != 64*num_blocks:
        print(f"Blocks {len(dc_coefficients)} found, expected {num_blocks}.")
        raise ValueError(f"Blocks {len(dc_coefficients)} found, expected {num_blocks}.")
    
    # Step 5: Reconstruct all 8x8 blocks from DC and AC coefficients
    reconstructed_blocks = []
    for i in range(len(dc_coefficients)):
        block_coefficients = np.zeros((block_size, block_size))
        block_coefficients.flat[zigzag_order[0]] = dc_coefficients[i]  # DC coefficient
        
        # Safely map AC coefficients using zigzag order
        block_ac = ac_coefficients[i * 63:(i + 1) * 63]
        

        block_coefficients.flat[zigzag_order[1:]] = block_ac
        dequantized_block = block_coefficients * Q_matrix
        reconstructed_block = idct(idct(dequantized_block.T, norm='ortho').T, norm='ortho')
        reconstructed_blocks.append(reconstructed_block)
    
    # Step 6: Check reconstructed_blocks alignment
    if len(reconstructed_blocks) != num_blocks:
        raise ValueError(f"Mismatch in block count: got {len(reconstructed_blocks)}, expected {num_blocks}")

    # Step 7: Combine blocks into the full image based on actual dimensions (height, width)
    reconstructed_image = np.zeros((height, width), dtype=np.float32)
    block_idx = 0

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            if block_idx < len(reconstructed_blocks):
                reconstructed_image[i:i + block_size, j:j + block_size] = reconstructed_blocks[block_idx]
                block_idx += 1
    
# Undo centering and clip intensity values to valid range [0, 255]
    reconstructed_image = np.clip(reconstructed_image + 128, 0, 255).astype(np.uint8)
    return reconstructed_image

def calculate_bpp(header, bitstream, image):
    bpp = sys.getsizeof(header)*8+len(bitstream)
    height, width = image.shape
    return bpp/(height*width)

def calculate_rmse(original, compressed):
    return np.sqrt(np.mean((original - compressed) ** 2))

def plot_images(im1, im2, str):
    combined_image = np.hstack((im1, im2))
    cv2.imshow("Original (Left) vs Compressed (Right)   " + str, combined_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def compress_image(image_path, Q):
    """
    Main function for JPEG compression preprocessing, DCT, and quantization.
    
    - Takes the image path as input from the user.
    - Accepts a quantization factor Q.
    - Preprocesses the grayscale image for JPEG compression.
    - Performs the DCT transformation on the image.
    - Quantizes the DCT coefficients and prepares them for encoding.
    """
    # Step 1: Get the image path from the user
    # image_path = "./data/im2.png"
    if not os.path.exists(image_path):
        print("Error: The specified image path does not exist.")
        return
    
    # Step 2: Get the quantization factor Q from the user
    try:
        if Q <= 0 or Q > 100:
            raise ValueError
    except ValueError:
        print("Error: Quantization factor Q should be an integer between 1 and 100.")
        return
    
    # Step 3: Preprocess the image
    try:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # print(f"Original Image Shape: {original_image.shape}")
        
        preprocessed_image = preprocess_grayscale_image(image_path)
        reconstructed_image = np.ones_like(preprocessed_image)
        image_height, image_width = preprocessed_image.shape
        input_image = np.array(preprocessed_image+128,dtype=np.uint8)
        # print("Image preprocessing complete!")
        # print(f"Preprocessed Image Shape: {preprocessed_image.shape}")
        
        # Step 4: Perform the DCT transformation on the preprocessed image
        # print("Performing DCT transformation...")
        dct_coefficients = perform_dct(preprocessed_image)
        # print(f"Number of DCT blocks: {len(dct_coefficients)}")
        
        # Step 5: Quantize the DCT coefficients
        # print(f"Quantizing DCT coefficients with Q = {Q}...")
        quantized_coefficients, Q_matrix = quantize_dct_blocks(dct_coefficients, Q)
        # print(f"Quantization complete. Total quantized blocks: {len(quantized_coefficients)}")

        # Prepare bitstream
        header, bitstream = prepare_bitstream(quantized_coefficients, Q_matrix)
        print(Q)
        header["image_dimensions"] = (image_height, image_width)
        # Display results
        # print("\n--- JPEG Compression Results ---")
        # print("Header Information:")
        # print(header)
        # print("\nCompressed Bitstream (First 200 bits):")
        # print(bitstream[:200])  # Show a snippet of the bitstream

        # print("\nCompressed Bitstream (Last 200 bits):")
        # print(bitstream[-200:])  # Show a snippet of the bitstream

        # print("\nCompression completed!")
        # Decode and reconstruct image
        reconstructed_image = decode_and_reconstruct(header, bitstream)
        
        rsme = calculate_rmse(input_image, reconstructed_image)
        bpp = calculate_bpp(header, bitstream, input_image)

        return rsme, bpp, input_image, reconstructed_image, header, bitstream
    except Exception as e:
        print(f"An error occurred: {e}")

def vary_q(image_path):
    quality_factors = range(10, 91, 10)
    rmse_values = []
    bpp_values = []
    images = []
    labels = ["Original Image"]
     
    for i in range(len(quality_factors)):
        rsme, bpp, input_image, reconstructed_image, header, bitstream = compress_image(image_path, quality_factors[i])
        # plot_images(input_image, reconstructed_image, f"Q: = {Q}")
        input_img = input_image
        images.append(reconstructed_image)
        rmse_values.append(rsme)
        bpp_values.append(bpp)
        labels.append(f"Q = {quality_factors[i]}")

        f1 = open(f"im{i}.kak","w")
        f1.write(str(header)+bitstream)
        f1.close()

    images.insert(0,input_img)
    # Create a figure and axes for the 2x5 grid
    fig1, axes = plt.subplots(2, 5, figsize=(15, 6))

    # Loop through the images and their corresponding labels
    for i, (ax, img, label) in enumerate(zip(axes.flatten(), images, labels)):
        # Load and display each image
        ax.imshow(img, cmap = "gray")
        ax.set_title(label, fontsize=10)
        ax.axis('off')  # Turn off axes for a cleaner look

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()
    
    fig2 = plt.figure()
    plt.plot(bpp_values, rmse_values, marker='o')
    plt.title("RMSE vs BPP")
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.grid()
    plt.show()

def main():
    for i in range(30,51):
        vary_q(f"./data/test{i}.png")


# Entry point
if __name__ == "__main__":
    main()


