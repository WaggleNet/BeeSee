# laptop_tcp_recv.py
import socket, struct
import cv2
import numpy as np

HOST = ''       # listen on all interfaces
PORT = 5555     # must match the Pi

# Set up server socket
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server.bind((HOST, PORT))
server.listen(1)
print(f"[Laptop] Listening on port {PORT}…")

conn, addr = server.accept()
print(f"[Laptop] Connection from {addr}")

data = b''
payload_size = struct.calcsize(">L")

# Read, decode, show loop
while True:
    # Read message length
    while len(data) < payload_size:
        packet = conn.recv(4096)
        if not packet:
            print("[Laptop] Connection closed by Pi")
            conn.close()
            server.close()
            cv2.destroyAllWindows()
            exit()
        data += packet
    packed_size = data[:payload_size]
    data = data[payload_size:]
    msg_size = struct.unpack(">L", packed_size)[0]

    # Read the frame data
    while len(data) < msg_size:
        data += conn.recv(4096)
    frame_data = data[:msg_size]
    data = data[msg_size:]

    # Decode JPEG and show
    frame = cv2.imdecode(
        np.frombuffer(frame_data, dtype=np.uint8),
        cv2.IMREAD_COLOR
    )

    

    cv2.imshow("Pi → Laptop Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

conn.close()
server.close()
cv2.destroyAllWindows()
