-- ============================================================
-- 푸른로봇 심포니 : 직관적인 빈피킹 TCP 스크립트 (함수 제거 버전)
-- ============================================================

---------------------------- 설정 ----------------------------
local SERVER_IP        = "192.168.1.101"   -- 좌표 서버 IP
local SERVER_PORT      = 20000

local APPROACH_DZ      = 80.0              -- 접근 높이 (mm)
local PLACE_POSE       = {400.0, -200.0, 150.0, 180.0, 0.0, 0.0}

local WS_MIN           = {-600.0, -600.0, 0.0}
local WS_MAX           = { 600.0,  600.0, 600.0}
local GRIP_WAIT_MS     = 500               -- 그리퍼 대기시간 (ms)

local MOVE_SPEED       = "30"              -- 이동 속도 (%)
local WORKFRAME        = 0
local TOOLFRAME        = 0
local GRIP_CLOSE_PORT  = 1                 -- 파지 DO 포트

local MAX_CYCLES       = 0                 -- 0 = 무한 반복
local MAX_ERRORS       = 3                 -- 연속 실패 허용 횟수

------------------------ TCP 통신 ------------------------
local cli = client(SERVER_IP, SERVER_PORT)

-- 1. TCP 서버 연결
if cli.open() ~= 0 then
    error("TCP 서버 연결 실패!")
end

local cycle, errors = 0, 0

---------------------------- 메인 반복 ----------------------------
while (MAX_CYCLES == 0 or cycle < MAX_CYCLES) do
    cycle = cycle + 1
    cli.flush()
    
    -- 2. 좌표 요청 (1byte 송신)
    cli.add_byte(1)
    if cli.send() == 0 and cli.recv(48) == 0 then
        -- 3. 좌표 수신 (double 6개)
        local ret, x  = cli.get_double()
        local ret, y  = cli.get_double()
        local ret, z  = cli.get_double()
        local ret, rx = cli.get_double()
        local ret, ry = cli.get_double()
        local ret, rz = cli.get_double()

        -- 4. 작업영역 검사
        local valid = (x >= WS_MIN[1] and x <= WS_MAX[1]) and
                      (y >= WS_MIN[2] and y <= WS_MAX[2]) and
                      (z >= WS_MIN[3] and z <= WS_MAX[3])

        if valid then
            print("cycle " .. cycle .. ": 피킹 동작 시작")
            errors = 0

            -- [1] 그리퍼 열기
            dio.reset(GRIP_CLOSE_PORT)

            -- [2] 피킹 상공 이동
            movel(pose(x, y, z + APPROACH_DZ, rx, ry, rz, WORKFRAME, TOOLFRAME), "b0", MOVE_SPEED)

            -- [3] 피킹 위치 하강
            movel(pose(x, y, z, rx, ry, rz, WORKFRAME, TOOLFRAME), "b0", MOVE_SPEED)

            -- [4] 그리퍼 파지 & 대기
            dio.set(GRIP_CLOSE_PORT)
            delay(GRIP_WAIT_MS)

            -- [5] 피킹 상공으로 상승
            movel(pose(x, y, z + APPROACH_DZ, rx, ry, rz, WORKFRAME, TOOLFRAME), "b0", MOVE_SPEED)

            -- [6] 플레이스 상공 이동
            movel(pose(PLACE_POSE[1], PLACE_POSE[2], PLACE_POSE[3] + APPROACH_DZ, PLACE_POSE[4], PLACE_POSE[5], PLACE_POSE[6], WORKFRAME, TOOLFRAME), "b0", MOVE_SPEED)

            -- [7] 플레이스 위치 하강
            movel(pose(PLACE_POSE[1], PLACE_POSE[2], PLACE_POSE[3], PLACE_POSE[4], PLACE_POSE[5], PLACE_POSE[6], WORKFRAME, TOOLFRAME), "b0", MOVE_SPEED)

            -- [8] 그리퍼 해제 & 대기
            dio.reset(GRIP_CLOSE_PORT)
            delay(GRIP_WAIT_MS)

            -- [9] 플레이스 상공으로 상승
            movel(pose(PLACE_POSE[1], PLACE_POSE[2], PLACE_POSE[3] + APPROACH_DZ, PLACE_POSE[4], PLACE_POSE[5], PLACE_POSE[6], WORKFRAME, TOOLFRAME), "b0", MOVE_SPEED)

        else
            errors = errors + 1
            print("cycle " .. cycle .. ": 작업영역 이탈 좌표 수신")
        end
    else
        errors = errors + 1
        print("cycle " .. cycle .. ": 통신 에러")
    end

    -- 연속 에러 누적 시 안전 종료
    if errors >= MAX_ERRORS then
        print("연속 오류 한도 초과로 종료합니다.")
        break
    end
end

cli.close()
