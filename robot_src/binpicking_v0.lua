-- ============================================================
-- 푸른로봇 심포니 : 경량화된 빈피킹 TCP 스크립트
-- ============================================================

---------------------------- 설정 ----------------------------
local SERVER_IP        = "192.168.1.101"   -- 좌표 서버 IP
local SERVER_PORT      = 20000

local APPROACH_DZ      = 80.0              -- 접근 높이 (mm)
local PLACE_POSE       = {400.0, -200.0, 150.0, 180.0, 0.0, 0.0}

local WS_MIN           = {-600.0, -600.0, 0.0}
local WS_MAX           = { 600.0,  600.0, 600.0}
local GRIP_WAIT_MS     = 500               -- 그리퍼 대기시간 (ms)[cite: 1]

local MOVE_SPEED       = "30"              -- 이동 속도 (%)
local WORKFRAME        = 0
local TOOLFRAME        = 0
local GRIP_CLOSE_PORT  = 1                 -- 파지 DO 포트[cite: 1]
local GRIP_OPEN_PORT   = nil               -- 단동 그리퍼 가정

local MAX_CYCLES       = 0                 -- 0 = 무한 반복
local MAX_ERRORS       = 3                 -- 연속 실패 허용 횟수

------------------ 로봇 동작 ------------------
local function move_linear(p)
    local target = pose(p[1], p[2], p[3], p[4], p[5], p[6], WORKFRAME, TOOLFRAME)
    movel(target, "b0", MOVE_SPEED) --
end

local function gripper(close)
    if close then
        if GRIP_OPEN_PORT then dio.reset(GRIP_OPEN_PORT) end --[cite: 1]
        dio.set(GRIP_CLOSE_PORT) --[cite: 1]
    else
        dio.reset(GRIP_CLOSE_PORT) --[cite: 1]
        if GRIP_OPEN_PORT then dio.set(GRIP_OPEN_PORT) end --[cite: 1]
    end
end

-------------------------- 유틸 --------------------------
local function above(p, dz)
    return {p[1], p[2], p[3] + dz, p[4], p[5], p[6]}
end

local function in_workspace(p)
    for i = 4, 6 do
        if p[i] < -180.0 or p[i] > 180.0 then return false end --
    end
    for i = 1, 3 do
        if p[i] < WS_MIN[i] or p[i] > WS_MAX[i] then return false end --
    end
    return true
end

------------------------ TCP 통신 ------------------------
local cli = client(SERVER_IP, SERVER_PORT) --[cite: 1]

local function connect()
    if cli.open() ~= 0 then --[cite: 1]
        print("서버 연결 실패")
        return false
    end
    cli.flush() --[cite: 1]
    return true
end

-- 좌표 요청 (요청 1byte 송신 -> double 6개=48byte 수신)[cite: 1]
local function request_pose()
    cli.flush() --[cite: 1]
    
    -- 1. 1바이트 커맨드 전송[cite: 1]
    cli.add_byte(1) --[cite: 1]
    if cli.send() ~= 0 then return nil, "송신 실패" end --[cite: 1]

    -- 2. 48바이트 (double * 6) 수신 대기[cite: 1]
    if cli.recv(48) ~= 0 then return nil, "수신 실패" end --[cite: 1]

    -- 3. 버퍼에서 좌표 데이터 꺼내기[cite: 1]
    local pick = {}
    for i = 1, 6 do
        local ret, val = cli.get_double() --[cite: 1]
        if ret ~= 0 then return nil, "데이터 수신 에러" end
        pick[i] = val
    end

    if not in_workspace(pick) then return nil, "작업영역 밖 좌표" end
    return pick
end

-------------------------- 시퀀스 --------------------------
local function pick_and_place(pick)
    gripper(false)
    move_linear(above(pick, APPROACH_DZ))
    move_linear(pick)
    gripper(true)
    delay(GRIP_WAIT_MS) --[cite: 1]
    move_linear(above(pick, APPROACH_DZ))

    move_linear(above(PLACE_POSE, APPROACH_DZ))
    move_linear(PLACE_POSE)
    gripper(false)
    delay(GRIP_WAIT_MS) --[cite: 1]
    move_linear(above(PLACE_POSE, APPROACH_DZ))
end

---------------------------- 메인 ----------------------------
if not in_workspace(PLACE_POSE) then
    error("PLACE_POSE 가 작업영역 밖입니다")
end

if connect() then
    local cycle, errors = 0, 0
    while (MAX_CYCLES == 0 or cycle < MAX_CYCLES) do
        cycle = cycle + 1
        local pick, why = request_pose()
        if pick then
            print("cycle " .. cycle .. " 피킹 완료")
            pick_and_place(pick)
            errors = 0
        else
            errors = errors + 1
            print("cycle " .. cycle .. " 실패(" .. errors .. "/" .. MAX_ERRORS .. "): " .. why)
            if errors >= MAX_ERRORS then
                print("연속 오류 한도 초과로 종료합니다.")
                break
            end
        end
    end
    cli.close() --[cite: 1]
end