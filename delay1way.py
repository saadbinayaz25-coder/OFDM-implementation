import zmq
import time
import struct

# ---------------- ZMQ Setup ----------------
context = zmq.Context()
socket = context.socket(zmq.PUSH)

PORT = "5555"
socket.connect(f"tcp://127.0.0.1:{PORT}")

time.sleep(1)  # ensure receiver is ready

# ---------------- Send Timestamp ----------------
t_send = time.time()

msg = struct.pack("d", t_send)
socket.send(msg)

print(f"Timestamp sent: {t_send:.9f}")
