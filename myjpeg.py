import numpy as np
from PIL import Image
from scipy.fftpack import dct, idct
import matplotlib.pyplot as plt
import heapq
from collections import defaultdict
import os

# Step 1: Divide the image into 8x8 blocks
def split_into_blocks(image, block_size=8):
    h, w = image.shape
    assert h % block_size == 0 and w % block_size == 0, "Image dimensions must be divisible by block size."
    blocks = image.reshape(h // block_size, block_size, -1, block_size).swapaxes(1, 2).reshape(-1, block_size, block_size)
    return blocks

# Step 2: Apply 2D DCT
def apply_dct(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

# Step 3: Apply Quantization
def quantize(block, quant_table):
    return np.round(block / quant_table).astype(int)

def dequantize(block, quant_table):
    return block * quant_table

# Step 4: Huffman Encoding
class HuffmanCoding:
    def __init__(self):
        self.heap = []
        self.codes = {}
        self.reverse_mapping = {}

    def build_frequency_dict(self, data):
        freq = defaultdict(int)
        for value in data:
            freq[value] += 1
        return freq

    def build_heap(self, frequency):
        for key in frequency:
            heapq.heappush(self.heap, (frequency[key], key))

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)
            merged = (node1[0] + node2[0], (node1, node2))
            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if isinstance(root[1], int):
            self.codes[root[1]] = current_code
            self.reverse_mapping[current_code] = root[1]
            return

        self.make_codes_helper(root[1][0], current_code + "0")
        self.make_codes_helper(root[1][1], current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        self.make_codes_helper(root, "")

    def encode(self, data):
        encoded_data = ''.join(self.codes[value] for value in data)
        return encoded_data

    def decode(self, encoded_data):
        current_code = ""
        decoded_data = []
        for bit in encoded_data:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_data.append(self.reverse_mapping[current_code])
                current_code = ""
        return decoded_data

# Step 5: File I/O
def save_to_file(encoded_data, quant_table, shape, filename):
    with open(filename, "wb") as f:
        # Save shape and quant table
        f.write(np.array(shape, dtype=np.int32).tobytes())
        f.write(quant_table.tobytes())
        # Save encoded data
        byte_data = int(encoded_data, 2).to_bytes((len(encoded_data) + 7) // 8, byteorder='big')
        f.write(byte_data)

def load_from_file(filename):
    with open(filename, "rb") as f:
        shape = np.frombuffer(f.read(8), dtype=np.int32)
        quant_table = np.frombuffer(f.read(64), dtype=np.int32).reshape(8, 8)
        encoded_data = f.read()
        bit_data = bin(int.from_bytes(encoded_data, byteorder='big'))[2:]
    return shape, quant_table, bit_data

# Step 6: Compress Image
def compress_image(image_path, quant_table, output_file):
    # Load and preprocess the image
    image = Image.open(image_path).convert("L")
    image = np.array(image)
    h, w = image.shape

    # Divide into blocks and apply DCT
    blocks = split_into_blocks(image)
    dct_blocks = np.array([apply_dct(block) for block in blocks])

    # Quantize the DCT coefficients
    quantized_blocks = np.array([quantize(block, quant_table) for block in dct_blocks])

    # Flatten and Huffman encode
    huffman = HuffmanCoding()
    flat_data = quantized_blocks.flatten()
    frequency = huffman.build_frequency_dict(flat_data)
    huffman.build_heap(frequency)
    huffman.merge_nodes()
    huffman.make_codes()
    encoded_data = huffman.encode(flat_data)

    # Save to file
    save_to_file(encoded_data, quant_table, (h, w), output_file)

# Step 7: Decompress Image
def decompress_image(input_file):
    shape, quant_table, encoded_data = load_from_file(input_file)

    # Decode Huffman data
    huffman = HuffmanCoding()
    huffman.build_heap(huffman.build_frequency_dict(encoded_data))
    huffman.merge_nodes()
    huffman.make_codes()
    decoded_data = np.array(huffman.decode(encoded_data))

    # Reshape and dequantize
    h, w = shape
    blocks = decoded_data.reshape(-1, 8, 8)
    dequantized_blocks = np.array([dequantize(block, quant_table) for block in blocks])

    # Apply inverse DCT
    idct_blocks = np.array([idct(idct(block.T, norm='ortho').T, norm='ortho') for block in dequantized_blocks])
    reconstructed_image = idct_blocks.reshape(h // 8, w // 8, 8, 8).swapaxes(1, 2).reshape(h, w)
    return np.clip(reconstructed_image, 0, 255).astype(np.uint8)

# Step 8: Evaluate RMSE and BPP
def evaluate(original_image, compressed_image, file_size):
    rmse = np.sqrt(np.mean((original_image - compressed_image) ** 2))
    bpp = file_size * 8 / (original_image.shape[0] * original_image.shape[1])
    return rmse, bpp

# Example Usage
if __name__ == "__main__":
    quant_table = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    # Compress the image
    compress_image("data/im1.png", quant_table, "compressed.bin")

    # Decompress the image
    reconstructed = decompress_image("compressed.bin")

    # Display the results
    original = np.array(Image.open("example.jpg").convert("L"))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(original, cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title("Reconstructed Image")
    plt.imshow(reconstructed, cmap="gray")
    plt.show()

    # Evaluate and print results
    file_size = os.path.getsize("compressed.bin")
    rmse, bpp = evaluate(original, reconstructed, file_size)
    print(f"RMSE: {rmse:.2f}, BPP: {bpp:.2f}")
