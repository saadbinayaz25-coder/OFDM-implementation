import numpy as np
import sys

def decode():
    if len(sys.argv) < 3:
        print("Usage: python decoder.py <input_file> <output_file>")
        return

    # Parity Check Matrix H
    H = np.array([
        [1, 1, 0, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 0, 0, 1]
    ])

    # Load coded bits
    coded_bits = np.fromfile(sys.argv[1], dtype=np.int8)
    blocks = coded_bits.reshape(-1, 7)
    
    # Calculate syndromes: S = r * H^T
    syndromes = np.dot(blocks, H.T) % 2
    
    corrected_blocks = blocks.copy()
    
    # Error Correction
    for i, s in enumerate(syndromes):
        if np.any(s):
            # Find column in H that matches the syndrome
            error_idx = np.where((H.T == s).all(axis=1))[0]
            if len(error_idx) > 0:
                idx = error_idx[0]
                corrected_blocks[i, idx] = 1 - corrected_blocks[i, idx]

    # Extract original 4 data bits (the first 4 columns)
    decoded_bits = corrected_blocks[:, :4].flatten()
    
    decoded_bits.astype(np.int8).tofile(sys.argv[2])
    print(f"Decoded {len(blocks)} blocks to {sys.argv[2]}")

if __name__ == "__main__":
    decode()