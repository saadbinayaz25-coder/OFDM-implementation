import zmq
import numpy as np

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://127.0.0.1:5557")

print("📡 Waiting for H_estim...")

# Receive metadata
meta = socket.recv_json()
shape = tuple(meta["shape"])
dtype = np.dtype(meta["dtype"])

# Receive raw bytes
data = socket.recv()

# Reconstruct numpy array
H_estim_rx = np.frombuffer(data, dtype=dtype).reshape(shape)

print("✅ H_estim received")
print("Shape :", H_estim_rx.shape)
print("values:", H_estim_rx[:102])
