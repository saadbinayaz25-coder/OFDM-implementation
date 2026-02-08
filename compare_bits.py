import numpy as np
import sys

if len(sys.argv) < 3:
    print(f"Usage: {sys.argv[0]} infile1 infile2")
    sys.exit(1)

data1 = np.fromfile(sys.argv[1], dtype='int8')
data2 = np.fromfile(sys.argv[2], dtype='int8')
n_errors = np.sum(np.abs(data1 - data2))
print(f"Files differ in {n_errors} locations.")