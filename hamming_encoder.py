import numpy as np
import sys

def encode():
    if len(sys.argv) < 3:
        print("Usage: python encoder.py <input_file> <output_file>")
        return

    # Generator Matrix G = [I | P]
    G = np.array([
        [1, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1]
    ])

    # Load bits from file
    raw_bits = np.fromfile(sys.argv[1], dtype=np.int8)
    
    # Pad with zeros if not a multiple of 4
    padding = (4 - len(raw_bits) % 4) % 4
    if padding:
        raw_bits = np.append(raw_bits, np.zeros(padding, dtype=np.int8))
    
    # Reshape to (N, 4) and encode
    data_blocks = raw_bits.reshape(-1, 4)
    coded_bits = np.dot(data_blocks, G) % 2
    
    # Flatten and save
    coded_bits.astype(np.int8).tofile(sys.argv[2])
    print(f"Encoded {len(data_blocks)} blocks to {sys.argv[2]}")

if __name__ == "__main__":
    encode()