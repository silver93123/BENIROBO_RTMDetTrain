-- ============================================================
-- Puloon Robot Symphony: Bin-Picking Script v5
-- 서버가 로봇 좌표(절대 좌표)로 변환해서 전송 -> 받은 좌표로 그대로 이동
-- 응답: "OK,<id>,X,Y,Z,Rx,Ry,Rz\n" (mm, deg) / "EMPTY\n" = 빈 비어있음(정상 종료)
-- ============================================================

---------------------------- Settings ----------------------------
local SERVER_IP        = "192.168.0.100"   -- Vision PC IP Address
local SERVER_PORT      = 5000
local REQUEST_STR      = "PICK\n"          -- UI Request Command
local EMPTY_TOKEN      = "EMPTY"           -- 빈 비어있음 응답 (서버와 합의)

-- Teaching Pendant [Data]-[Pose]
local WAIT_POSE        = var.p(5)          -- Camera Capture Point (P5)
local PLACE_POSE       = var.p(2)          -- Place Approach Position (P2): 놓는 면 위 APPROACH_DZ 높이로 교시

local WAIT_SETTLE_MS   = 1000              -- 촬영 전 정지 시간 (ms, 진동 안정화)

local APPROACH_DZ      = 150.0              -- Approach Height (mm)
local RETREAT_VIA_WAIT = true              -- 파지 후 P5 경유해서 P2로 이동 (벽/부품 충돌 방지)

-- 서버가 좌표를 계산한 기준과 일치해야 함
local WORKFRAME        = 0                 -- 워크 좌표계 번호 (0~10)
local TOOLFRAME        = 0                 -- 그리퍼 끝(TCP)이 등록된 툴 좌표계 번호 (0~10)

-- Workspace Boundary Limits (절대 좌표, mm)
local WS_MIN_X         = -600.0
local WS_MAX_X         =  600.0
local WS_MIN_Y         = -600.0
local WS_MAX_Y         =  600.0
local WS_MIN_Z         =    0.0            -- 실제 바닥/빈 바닥 안전 하한으로 수정할 것
local WS_MAX_Z         =  600.0

local GRIP_WAIT_MS     = 500               -- Gripper Delay (ms)

local MOVE_SPEED       = "s30"             -- Motion Speed (%)
local GRIP_CLOSE_PORT  = 1                 -- Digital Output Port

local MAX_CYCLES       = 3                 -- Max Cycle Count (0: Infinite)
local MAX_ERRORS       = 3                 -- Max Consecutive Error Count

local RECV_MAX_TRIES   = 100
local RECV_POLL_MS     = 100

-- Vision Grasping Offset (mm)
local offset_x         = 0.0
local offset_y         = 0.0
local offset_z         = 30.0

local FIXED_RX       = -178.96
local FIXED_RY       = -1.45
local FIXED_RZ       = 131.82
local FIXED_Z       = 582.17

------------------------ TCP Connection ------------------------
local cli = client(SERVER_IP, SERVER_PORT)

-- 1. Connect to TCP Server
local open_ret = cli.open()
if open_ret ~= 0 then
    msg.error("TCP failed: " .. tostring(open_ret))
    return
end

local cycle, errors = 0, 0

------------------------ Vision Request ------------------------
-- 수신 버퍼 앞에 붙는 쓰레기 바이트(비문자 24바이트 등)를 건너뛰기 위해,
-- "OK," / "NG," / EMPTY 가 처음 나오는 위치를 찾는다. 없으면 nil
local function find_reply_start(str)
    local tags = { "OK,", "NG,", EMPTY_TOKEN }
    local start = nil
    for t = 1, #tags do
        local p = string.find(str, tags[t], 1, true)
        if p and (start == nil or p < start) then start = p end
    end
    return start
end

-- 좌표 요청 -> 수신 -> 파싱 -> 검사
-- 반환: "ok",    nil,     x, y, z, rx, ry, rz
--       "empty", nil
--       "error", 사유(string)
local function request_pick()
    -- Flush Buffer & Send Request
    cli.flush()
    local send_ret = cli.write(REQUEST_STR)
    if send_ret ~= 0 then
        return "error", "send failed: " .. tostring(send_ret)
    end

    -- Receive one line (until newline)
    local NL = string.char(10)
    local res_str = ""
    for i = 1, RECV_MAX_TRIES do
        local _, s = cli.read(128, 1)
        if s ~= nil and s ~= "" then
            res_str = res_str .. s
            local st = find_reply_start(res_str)
            if st and string.find(res_str, NL, st, true) then break end
        else
            delay(RECV_POLL_MS)
        end
    end

    local start = find_reply_start(res_str)
    if not start then
        return "error", "no OK/NG/EMPTY in reply (len=" .. tostring(#res_str) .. "): [" .. res_str .. "]"
    end
    local garbage = start - 1                 -- 앞에 붙은 쓰레기 바이트 수
    res_str = string.sub(res_str, start)
    local nl = string.find(res_str, NL, 1, true)
    if not nl then
        return "error", "timeout (no newline): [" .. res_str .. "]"
    end
    res_str = string.sub(res_str, 1, nl - 1)
    res_str = string.gsub(res_str, string.char(13), "")
    res_str = string.gsub(res_str, "%s+", "")
    msg.info("Recv Data: [" .. tostring(res_str) .. "] len=" .. tostring(#res_str) ..
             " (앞 쓰레기 " .. tostring(garbage) .. "바이트 제거)")

    -- Split by ","
    local tokens = {}
    local pos = 1
    while true do
        local p = string.find(res_str, ",", pos, true)
        if not p then
            tokens[#tokens + 1] = string.sub(res_str, pos)
            break
        end
        tokens[#tokens + 1] = string.sub(res_str, pos, p - 1)
        pos = p + 1
    end

    -- Validate "OK" and Token Count (OK,id,X,Y,Z,Rx,Ry,Rz -> Min 8 tokens)
    if tokens[1] == EMPTY_TOKEN then
        return "empty"
    end
    if tokens[1] ~= "OK" or #tokens < 8 then
        return "error", "vision failed: " .. res_str
    end

    local x  = tonumber(tokens[3])
    local y  = tonumber(tokens[4])
    local z  = tonumber(tokens[5])
    local rx = tonumber(tokens[6])
    local ry = tonumber(tokens[7])
    local rz = tonumber(tokens[8])

    -- 수신 좌표 검사 (절대 좌표 + 접근점)
    if not (x and y and z and rx and ry and rz) then
        return "error", "non-numeric value: " .. res_str
    end
    if x ~= x or y ~= y or z ~= z or rx ~= rx or ry ~= ry or rz ~= rz then
        return "error", "NaN value: " .. res_str
    end
    if x < WS_MIN_X or x > WS_MAX_X then return "error", "X out of range: " .. tostring(x) end
    if y < WS_MIN_Y or y > WS_MAX_Y then return "error", "Y out of range: " .. tostring(y) end
    if z < WS_MIN_Z or (z + APPROACH_DZ) > WS_MAX_Z then return "error", "Z out of range: " .. tostring(z) end
    if rx < -180.0 or rx > 180.0 or ry < -180.0 or ry > 180.0 or rz < -180.0 or rz > 180.0 then
        return "error", "Rx/Ry/Rz out of range: " .. res_str
    end

    return "ok", nil, x, y, z, rx, ry, rz
end

---------------------------- Main Sequence ----------------------------
while (MAX_CYCLES == 0 or cycle < MAX_CYCLES) do
    cycle = cycle + 1

    -- [1] Move to Wait/Capture Point (P5) & Settle
    movel(WAIT_POSE, "b0", MOVE_SPEED)
    delay(WAIT_SETTLE_MS)

    -- [2] Request Pick Coordinates (Vision)
    local status, err_msg, x, y, z, rx, ry, rz = request_pick()

    if status == "empty" then
        msg.info("Bin empty - finished")
        break
    elseif status == "error" then
        errors = errors + 1
        msg.error("cycle " .. tostring(cycle) .. ": " .. tostring(err_msg))
    else
        errors = 0

        pick_dz = -APPROACH_DZ + offset_z

        local P_APPROACH = pose(x + offset_x, y + offset_y, z + APPROACH_DZ, FIXED_RX, FIXED_RY, FIXED_RZ)
        local P_PICK     = pose(x + offset_x, y + offset_y, z + offset_z, FIXED_RX, ry, rz,)

        -- 1. Open Gripper
        dio.reset(GRIP_CLOSE_PORT)

        -- 2. Move to Pick Approach Position
        movel(P_APPROACH, "b0", MOVE_SPEED)

        -- 3. Move down to Pick Position

        toolframe(1)
        tool.rotz(rz, MOVE_SPEED)
        transz(pick_dz, MOVE_SPEED)

        -- 4. Close Gripper & Delay
        dio.set(GRIP_CLOSE_PORT)
        delay(GRIP_WAIT_MS)

        -- 5. Move back to Pick Approach Position
        movel(P_APPROACH, "b0", MOVE_SPEED)


        -- 7. Move to Place Approach Position (P2)
        movep(PLACE_POSE, "b0", MOVE_SPEED)

        -- 8. Move down to Place Position
        transz(-APPROACH_DZ, MOVE_SPEED)

        -- 9. Open Gripper & Delay
        dio.reset(GRIP_CLOSE_PORT)
        delay(GRIP_WAIT_MS)

        -- 10. Move back to Place Approach Position
        transz(APPROACH_DZ, MOVE_SPEED)
    end

    -- Stop Robot if Consecutive Errors Exceed Limit
    if errors >= MAX_ERRORS then
        msg.error("Max error limit reached: " .. tostring(MAX_ERRORS))
        break
    end
end

-- Return to Wait Position (P5) and Close Client
movel(WAIT_POSE, "b0", MOVE_SPEED)
cli.close()