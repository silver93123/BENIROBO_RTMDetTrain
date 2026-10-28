-- ============================================================
-- Puloon Robot Symphony: Bin-Picking Script (English Logs)
-- ============================================================

---------------------------- Settings ----------------------------
local SERVER_IP        = "192.168.0.100"   -- Vision PC IP Address
local SERVER_PORT      = 5000              -- UI Setting Port: 5000
local REQUEST_STR      = "PICK\n"          -- UI Request Command

-- Teaching Pendant [Data]-[Pose] P1, P2
local WAIT_POSE        = var.p(1)          -- Camera Capture Point (P1)
local PLACE_POSE       = var.p(2)          -- Place Position (P2)[cite: 6]

local APPROACH_DZ      = 80.0              -- Approach Height (mm)

-- Workspace Boundary Limits
local WS_MIN_X         = -600.0
local WS_MAX_X         =  600.0
local WS_MIN_Y         = -600.0
local WS_MAX_Y         =  600.0
local WS_MIN_Z         =    0.0
local WS_MAX_Z         =  600.0

local GRIP_WAIT_MS     = 500               -- Gripper Delay (ms)

local MOVE_SPEED       = "s30"             -- Motion Speed (%)
local WORKFRAME        = 0
local TOOLFRAME        = 0
local GRIP_CLOSE_PORT  = 1                 -- Digital Output Port

local MAX_CYCLES       = 3                 -- Max Cycle Count (0: Infinite)
local MAX_ERRORS       = 3                 -- Max Consecutive Error Count

-- Reply wait: (RECV_MAX_TRIES x (read time + RECV_POLL_MS)) is the longest wait for the vision reply.
-- The vision PC needs about 0.4-2 s per request (first request after server start can be longer).
local RECV_MAX_TRIES   = 100
local RECV_POLL_MS     = 100

-- Vision Grasping Offset (mm)
local offset_x         = 0.0
local offset_y         = 0.0
local offset_z         = 10.0

------------------------ TCP Connection ------------------------
local cli = client("192.168.0.100", 5000)

-- 1. Connect to TCP Server
local open_ret = cli.open()
if open_ret ~= 0 then
    msg.error("TCP failed: " .. tostring(open_ret))
    return
end

local cycle, errors = 0, 0

---------------------------- Main Sequence ----------------------------
while (MAX_CYCLES == 0 or cycle < MAX_CYCLES) do
    cycle = cycle + 1
    
    -- [1] Move to Wait/Capture Point (P1)
    movel(WAIT_POSE, "b0", MOVE_SPEED)
    
    -- [2] Flush Buffer & Send Request
    cli.flush()
    
    local send_ret = cli.write(REQUEST_STR)
    
    -- NOTE: cli.recv(n) consumes the first n bytes of the reply (they are lost), so it is NOT used here.
    -- Poll cli.read() until the end-of-line character (code 10) arrives or the wait limit is reached.
    local NL = string.char(10)
    local res_str = ""
    for i = 1, RECV_MAX_TRIES do
        local _, s = cli.read(128, 1)
        if s ~= nil and s ~= "" then
            res_str = res_str .. s
        if string.find(res_str, NL, 1, true) then break end
        else
            delay(RECV_POLL_MS)
        end
    end
    -- Keep only the first line: anything after the end-of-line character is junk (stale buffer bytes etc.).
    local nl = string.find(res_str, NL, 1, true)
if nl then res_str = string.sub(res_str, 1, nl - 1) end
    res_str = string.gsub(res_str, string.char(13), "")
    local read_ret = 0
if res_str == "" then read_ret = -1 end   -- nothing received within the wait limit
    
    if read_ret ~= 0 then
        msg.warn("cycle " .. tostring(cycle) .. ": timeout (code: " .. tostring(read_ret) .. ")")
        errors = errors + 1
    elseif res_str == nil or res_str == "" then
        msg.warn("cycle " .. tostring(cycle) .. ": empty data received")
        errors = errors + 1
    else
        -- [3] 수신 데이터 팝업 표시 (실제 전달받은 원본 데이터 확인용)
        msg.info("Recv Data: [" .. tostring(res_str) .. "] len=" .. tostring(#res_str))
        
        -- 백슬래시(\) 없이 개행문자 및 공백 정제 (%s 사용)
        res_str = string.gsub(res_str, "%s+", "")
        
        -- 쉼표(,) 기준 순수 문자열 분할
        local tokens = {}
        local str = res_str
        local pos = 1
        while true do
            local p = string.find(str, ",", pos, true)
            if not p then
                tokens[#tokens + 1] = string.sub(str, pos)
                break
            end
            tokens[#tokens + 1] = string.sub(str, pos, p - 1)
            pos = p + 1
        end
        
        -- Validate "OK" and Token Count (OK,1,X,Y,Z,Rx,Ry,Rz -> Min 8 tokens)
        if tokens[1] == "OK" and #tokens >= 8 then
            local x  = tonumber(tokens[3]) + offset_x
            local y  = tonumber(tokens[4]) + offset_y
            local z  = tonumber(tokens[5]) + offset_z
            local rx = tonumber(tokens[6])
            local ry = tonumber(tokens[7])
            local rz = tonumber(tokens[8])
            
            -- [4] Validate Workspace & Coordinates
            if x and y and z and rx and ry and rz then
                if x >= WS_MIN_X and x <= WS_MAX_X and y >= WS_MIN_Y and y <= WS_MAX_Y and z >= WS_MIN_Z and z <= WS_MAX_Z then
                    errors = 0
                    
                    -- 1. Open Gripper
                    dio.reset(GRIP_CLOSE_PORT)
                    
                    -- 2. Move to Pick Approach Position
                    movep(pose(x, y, z + APPROACH_DZ, rx, ry, rz), "b0", MOVE_SPEED)
                    
                    -- 3. Move down to Pick Position
                    movep(pose(x, y, z, rx, ry, rz), "b0", MOVE_SPEED)
                    
                    -- 4. Close Gripper & Delay
                    dio.set(GRIP_CLOSE_PORT)
                    delay(GRIP_WAIT_MS)
                    
                    -- 5. Move back to Pick Approach Position
                    movel(pose(x, y, z + APPROACH_DZ, rx, ry, rz), "b0", MOVE_SPEED)
                    
                    -- 6. Move to Place Approach Position using transz
                    movel(PLACE_POSE, "b0", MOVE_SPEED)
                    transz(APPROACH_DZ)
                    
                    -- 7. Move down to Place Position (P2)
                    movel(PLACE_POSE, "b0", MOVE_SPEED)
                    
                    -- 8. Open Gripper & Delay
                    dio.reset(GRIP_CLOSE_PORT)
                    delay(GRIP_WAIT_MS)
                    
                    -- 9. Move back to Place Approach Position
                    transz(APPROACH_DZ)
                    
                else
                    errors = errors + 1
                    msg.error("Out of Workspace! X:" .. tostring(x) .. " Y:" .. tostring(y) .. " Z:" .. tostring(z))
                end
            else
                errors = errors + 1
                msg.error("Value conversion failed: " .. tostring(res_str))
            end
        else
            errors = errors + 1
            msg.error("vision failed: " .. tostring(res_str))
        end
    end
    
    -- Stop Robot if Consecutive Errors Exceed Limit
    if errors >= MAX_ERRORS then
        msg.error("Max error limit reached: " .. tostring(MAX_ERRORS))
        break
    end
end

-- Return to Wait Position (P1) and Close Client
movel(WAIT_POSE, "b0", MOVE_SPEED)
cli.close()