import zmq
import struct
import time

context = zmq.Context()
socket = context.socket(zmq.PULL)
port = "5555"
socket.bind(f"tcp://*:{port}")
print("ZMQ PULL sink started, waiting for data...")
msg = socket.recv()
rx_time = time.time()
tx_time = struct.unpack('d', msg)[0]
delay = rx_time - tx_time

print(f"TX time : {tx_time:.6f}")

print(f"RX time : {rx_time:.6f}")

print(f"Delay : {delay * 1000:.3f} ms")

socket.close()
context.term()





''''import zmq
import time
import struct

# ---------------- ZMQ Setup ----------------
context = zmq.Context()
socket = context.socket(zmq.PUSH)

SINK_IP = "192.168.10.2"   # Receiver IP
PORT = "5555"

socket.connect(f"tcp://{SINK_IP}:{PORT}")

# Allow receiver to connect
time.sleep(1)

# ---------------- Send ONCE ----------------
tx_time = time.time()                  # current Unix time
msg = struct.pack('d', tx_time)        # pack as double (8 bytes)

socket.send(msg)
print(f"Sent time: {tx_time:.6f}")

socket.close()
context.term() '''
