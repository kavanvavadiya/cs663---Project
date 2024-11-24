import numpy as np
import cv2
from scipy.fftpack import dct, idct
from matplotlib import pyplot as plt
import heapq
from collections import defaultdict
import os

# Helper Functions
def compute_dct_2d(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def compute_idct_2d(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

def quantize_block(block, quant_matrix):
    return np.round(block / quant_matrix).astype(int)

def dequantize_block(block, quant_matrix):
    return (block * quant_matrix).astype(float)

# Huffman Tree
class HuffmanNode:
    def __init__(self, symbol=None, freq=0):
        self.symbol = symbol
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(symbol, freq) for symbol, freq in freq_dict.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(freq=node1.freq + node2.freq)
        merged.left, merged.right = node1, node2
        heapq.heappush(heap, merged)
    
    return heap[0]

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

def huffman_encode(data, codes):
    return "".join(codes[symbol] for symbol in data)

def huffman_decode(encoded_data, tree):
    decoded_data = []
    node = tree
    for bit in encoded_data:
        node = node.left if bit == '0' else node.right
        if node.symbol is not None:
            decoded_data.append(node.symbol)
            node = tree
    return decoded_data

# Main Processing Functions
def compress_image(image, quant_matrix):
    h, w = image.shape
    blocks = []
    quantized_blocks = []
    
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            block = image[i:i+8, j:j+8]
            dct_block = compute_dct_2d(block - 128)  # Shift pixel range to [-128, 127]
            quant_block = quantize_block(dct_block, quant_matrix)
            blocks.append(quant_block)
            quantized_blocks.append(quant_block.flatten())
    
    # Flatten all quantized coefficients and build Huffman tree
    all_coefficients = np.concatenate(quantized_blocks)
    freq_dict = defaultdict(int)
    for coeff in all_coefficients:
        freq_dict[coeff] += 1
    
    huffman_tree = build_huffman_tree(freq_dict)
    huffman_codes = build_huffman_codes(huffman_tree)
    encoded_data = huffman_encode(all_coefficients, huffman_codes)
    
    return huffman_tree, encoded_data

def decompress_image(blocks, quant_matrix, h, w):
    reconstructed_image = np.zeros((h, w), dtype=np.uint8)
    idx = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            quant_block = blocks[idx]
            idct_block = compute_idct_2d(dequantize_block(quant_block, quant_matrix))
            reconstructed_image[i:i+8, j:j+8] = np.clip(idct_block + 128, 0, 255)
            idx += 1
    return reconstructed_image

# RMSE and BPP Calculation
def calculate_rmse(original, compressed):
    return np.sqrt(np.mean((original - compressed) ** 2))

def calculate_bpp(encoded_data, image_shape):
    total_bits = len(encoded_data)
    num_pixels = image_shape[0] * image_shape[1]
    return total_bits / num_pixels

# Main Execution
def main():
    # Load image
    image = cv2.imread("data/im1.png", cv2.IMREAD_GRAYSCALE)
    h, w = image.shape
    
    # Define quantization matrix for Q=50
    Tb = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ])
    
    quality_factors = range(10, 101, 10)
    rmse_values = []
    bpp_values = []
    
    for Q in quality_factors:
        S = 5000 / Q if Q < 50 else 200 - 2 * Q
        quant_matrix = np.floor((S * Tb + 50) / 100)
        quant_matrix[quant_matrix == 0] = 1
        blocks = []
        # Compress and decompress image
        huffman_tree, encoded_data = compress_image(image, quant_matrix)
        decoded_data = huffman_decode(encoded_data=encoded_data,tree=huffman_tree)
        
        for i in range(int(len(decoded_data)/64)):
            blocks.append(np.reshape(decoded_data[i*64:(i+1)*64],(8,8)))
        decompressed_image = decompress_image(blocks, quant_matrix, h, w)
        
        # Calculate RMSE and BPP
        rmse = calculate_rmse(image, decompressed_image)
        bpp = calculate_bpp(encoded_data, image.shape)
        
        rmse_values.append(rmse)
        bpp_values.append(bpp)

        # Visual comparison for each Q
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title(f"Original Image (Q={Q})")
        plt.imshow(image, cmap="gray")
        plt.axis("off")
        plt.subplot(1, 2, 2)
        plt.title(f"Decompressed Image (Q={Q})")
        plt.imshow(decompressed_image, cmap="gray")
        plt.axis("off")
        plt.show()
    
    # Plot RMSE vs BPP
    plt.figure()
    plt.plot(bpp_values, rmse_values, marker='o')
    plt.title("RMSE vs BPP")
    plt.xlabel("Bits Per Pixel (BPP)")
    plt.ylabel("Root Mean Squared Error (RMSE)")
    plt.grid()
    plt.show()




main()
