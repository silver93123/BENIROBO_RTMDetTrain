import socket
s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('0.0.0.0', 5000))  # 포트 번호 확인
s.listen(1)
print("=== PC 포트 5000번 접속 대기 중... ===")
conn, addr = s.accept()
print(f"로봇 접속 성공!: {addr}")