import socket

# 비전 UI 설정과 동일하게 포트 5000번 오픈
HOST = '0.0.0.0'
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

print(f"=== TCP 테스트 서버가 포트 {PORT}에서 실행 중입니다 ===")
print("로봇 접속을 기다리는 중...")

while True:
    try:
        client_socket, addr = server_socket.accept()
        print(f"\n[+] 로봇 접속됨: {addr[0]}:{addr[1]}")
        
        while True:
            data = client_socket.recv(1024)
            if not data:
                print("[-] 로봇 연결 종료됨")
                break
            
            req = data.decode('utf-8', errors='ignore').strip()
            print(f"[수신데이터] 로봇 -> PC: '{req}'")
            
            # 로봇이 "PICK" 요청을 보내면 비전 결과 가짜 좌표 응답 전송
            if "PICK" in req:
                # OK,1,X,Y,Z,Rx,Ry,Rz\n 형태 응답
                dummy_response = "OK,1,-63.40,-29.20,546.00,39.60,21.40,52.10\n"
                client_socket.sendall(dummy_response.encode('utf-8'))
                print(f"[송신응답] PC -> 로봇: {dummy_response.strip()}")
            else:
                client_socket.sendall(b"NG,UNKNOWN_CMD\n")
                
        client_socket.close()
    except Exception as e:
        print(f"오류 발생: {e}")
EOF