import numpy as np
import sys

def generate_bits():
    if len(sys.argv) < 3:
        print("Usage: python generate_bits.py <num_bits> <output_file>")
        return

    try:
        num_bits = int(sys.argv[1])
    except ValueError:
        print("Error: <num_bits> must be an integer.")
        return

    output_file = sys.argv[2]

    # Generate random bits (0 or 1) with equal probability
    bits = np.random.randint(0, 2, size=num_bits, dtype=np.int8)

    # Save to file
    bits.tofile(output_file)
    print(f"Successfully generated {num_bits} bits and saved to '{output_file}'.")

if __name__ == "__main__":
    generate_bits()