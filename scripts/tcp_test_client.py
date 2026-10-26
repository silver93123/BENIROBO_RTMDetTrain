"""5번 탭(ICP 정합 TCP 서버) 점검용 테스트 클라이언트 - 로봇 없이 서버가 제대로 응답하는지 확인한다.

사용 예 (앱에서 '서버 시작'을 누른 뒤):

    python scripts/tcp_test_client.py                       # 127.0.0.1:5000 에 PICK 1회
    python scripts/tcp_test_client.py --ping                # 연결 확인만 (카메라/모델 안 건드림)
    python scripts/tcp_test_client.py --host 192.168.0.10 --port 6000 --repeat 5 --interval 1.0
    python scripts/tcp_test_client.py --cmd GETPOS --sep ";"   # 설정 탭에서 명령/구분자를 바꿨다면 맞춰서

응답 형식 (설정 탭 > TCP 통신 의 기본값):
    OK,<n>,x,y,z[,roll,pitch,yaw] x n     성공. n개 후보가 정렬된 순서(앞이 우선)로 들어 있다.
    NG,<코드>                             실패 (NO_DETECTION, NO_VALID_POSE, CAMERA_ERROR, BUSY ...)
    좌표는 카메라 좌표계 mm / deg.

설정 탭에서 '각 후보에 roll/pitch/yaw 포함'을 껐다면 --no-rpy 를 주세요.
"""
from __future__ import annotations

import argparse
import socket
import sys
import time


def parse_response(text: str, sep: str = ",", ok_prefix: str = "OK", ng_prefix: str = "NG",
                   include_rpy: bool = True) -> dict:
    """응답 한 줄을 파싱한다. 형식이 맞지 않으면 ValueError.

    반환: {"ok": True, "picks": [{"xyz": (x,y,z), "rpy": (r,p,y) | None}, ...]}
          {"ok": False, "code": "NO_DETECTION"}
    """
    fields = text.strip().split(sep)
    if fields[0] == ng_prefix:
        return {"ok": False, "code": fields[1] if len(fields) > 1 else ""}
    if fields[0] != ok_prefix:
        raise ValueError(f"알 수 없는 응답: {text!r}")
    n = int(fields[1])
    per = 6 if include_rpy else 3
    values = fields[2:]
    if len(values) != n * per:
        raise ValueError(f"필드 개수가 맞지 않습니다: 후보 {n}개 x {per} = {n * per} 기대, 실제 {len(values)} ({text!r})")
    picks = []
    for i in range(n):
        v = [float(x) for x in values[i * per:(i + 1) * per]]
        picks.append({"xyz": tuple(v[:3]), "rpy": tuple(v[3:6]) if include_rpy else None})
    return {"ok": True, "picks": picks}


def request(host: str, port: int, cmd: str, terminator: bytes, timeout: float) -> str:
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.settimeout(timeout)
        s.sendall(cmd.encode("utf-8") + terminator)
        buf = b""
        while not (buf.endswith(b"\n") or buf.endswith(b"\r")):
            chunk = s.recv(4096)
            if not chunk:
                break
            buf += chunk
    return buf.decode("utf-8", errors="replace").strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--cmd", default="PICK", help="좌표 요청 명령 (설정 탭 값과 같게)")
    ap.add_argument("--ping", action="store_true", help="PICK 대신 PING으로 연결만 확인")
    ap.add_argument("--ping-cmd", default="PING")
    ap.add_argument("--sep", default=",")
    ap.add_argument("--ok", default="OK")
    ap.add_argument("--ng", default="NG")
    ap.add_argument("--no-rpy", action="store_true")
    ap.add_argument("--term", choices=["LF", "CRLF", "CR", "NONE"], default="LF", help="요청 끝 문자")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--interval", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=60.0, help="응답 대기 시간(초). 촬영+검출+정합 시간을 감안")
    args = ap.parse_args()

    term = {"LF": b"\n", "CRLF": b"\r\n", "CR": b"\r", "NONE": b""}[args.term]
    cmd = args.ping_cmd if args.ping else args.cmd
    rc = 0
    for i in range(args.repeat):
        t0 = time.perf_counter()
        try:
            text = request(args.host, args.port, cmd, term, args.timeout)
        except OSError as exc:
            print(f"[{i + 1}] 접속/통신 실패: {exc}")
            return 2
        dt = (time.perf_counter() - t0) * 1000.0
        print(f"[{i + 1}] {cmd} -> {text}   ({dt:.0f} ms)")
        if not args.ping:
            try:
                r = parse_response(text, args.sep, args.ok, args.ng, not args.no_rpy)
            except ValueError as exc:
                print(f"    파싱 실패: {exc}")
                rc = 1
            else:
                if r["ok"]:
                    for k, p in enumerate(r["picks"], 1):
                        rpy = f"  roll/pitch/yaw = {p['rpy']}" if p["rpy"] else ""
                        print(f"    후보 {k}: x/y/z = {p['xyz']} mm{rpy}")
                else:
                    print(f"    실패 코드: {r['code']}")
        if i + 1 < args.repeat:
            time.sleep(args.interval)
    return rc


if __name__ == "__main__":
    sys.exit(main())
