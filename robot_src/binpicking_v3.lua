-- ============================================================
-- Puloon Robot Symphony: Bin-Picking Script (Standard Motion)
-- ============================================================

---------------------------- Settings ----------------------------
local SERVER_IP        = "192.168.0.100"   -- Vision PC IP Address
local SERVER_PORT      = 5000              -- UI Setting Port: 5000
local REQUEST_STR      = "PICK\n"          -- UI Request Command

-- Teaching Pendant [Data]-[Pose] P1, P2
local WAIT_POSE        = var.p(1)          -- Camera Capture Point (P1)
local PLACE_POSE       = var.p(2)          -- Place Position (P2)

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
local GRIP_CLOSE_PORT  = 1                 -- Digital Output Port

local MAX_CYCLES       = 3                 -- Max Cycle Count (0: Infinite)
local MAX_ERRORS       = 3                 -- Max Consecutive Error Count

local RECV_MAX_TRIES   = 100
local RECV_POLL_MS     = 100

-- Vision Grasping Offset (mm)
local offset_x         = 0.0
local offset_y         = 80.0
local offset_z         = 180.0

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

    local nl = string.find(res_str, NL, 1, true)
    if nl then res_str = string.sub(res_str, 1, nl - 1) end
    res_str = string.gsub(res_str, string.char(13), "")
    local read_ret = 0
    if res_str == "" then read_ret = -1 end
    
    if read_ret ~= 0 then
        msg.warn("cycle " .. tostring(cycle) .. ": timeout (code: " .. tostring(read_ret) .. ")")
        errors = errors + 1
    elseif res_str == nil or res_str == "" then
        msg.warn("cycle " .. tostring(cycle) .. ": empty data received")
        errors = errors + 1
    else
        msg.info("Recv Data: [" .. tostring(res_str) .. "] len=" .. tostring(#res_str))
        
        res_str = string.gsub(res_str, "%s+", "")
        
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
            local cam_dx = tonumber(tokens[3])
            local cam_dy = tonumber(tokens[4])
            local cam_dz = tonumber(tokens[5])
            local rx     = tonumber(tokens[6])
            local ry     = tonumber(tokens[7])
            local rz     = tonumber(tokens[8])
            
            if cam_dx and cam_dy and cam_dz and rx and ry and rz then
                -- [3] 상대 변위 계산 (축 매핑 + 그리퍼 오프셋 보정)
                local rel_x = - cam_dy + offset_x
                local rel_y = - cam_dx + offset_y
                local rel_z = - cam_dz + offset_z
                
                errors = 0
                
                -- 1. Open Gripper
                dio.reset(GRIP_CLOSE_PORT)
                
                -- 2. Move to Pick Approach Position (P1 기준 상대 이동)
                trans(rel_x, rel_y, rel_z + APPROACH_DZ, MOVE_SPEED)
                
                -- 3. Rz 회전 변위 적용 (0이 아닐 경우)
                if rz ~= 0 then
                    rotz(rz, MOVE_SPEED)
                end
                
                -- 4. Move down to Pick Position (Z축 하강)
                transz(-APPROACH_DZ, MOVE_SPEED)
                
                -- 5. Close Gripper & Delay
                dio.set(GRIP_CLOSE_PORT)
                delay(GRIP_WAIT_MS)
                
                -- 6. Move back to Pick Approach Position (Z축 복귀)
                transz(APPROACH_DZ, MOVE_SPEED)
                
                -- 7. Rz 회전 복원
                if rz ~= 0 then
                    rotz(-rz, MOVE_SPEED)
                end
                
                -- 8. Move to Place Position (P2)
                movep(PLACE_POSE, "b0", MOVE_SPEED)
                
                -- 9. Move down to Place Position
                transz(-APPROACH_DZ, MOVE_SPEED)
                
                -- 10. Open Gripper & Delay
                dio.reset(GRIP_CLOSE_PORT)
                delay(GRIP_WAIT_MS)
                
                -- 11. Move back to Place Approach Position
                transz(APPROACH_DZ, MOVE_SPEED)
                
            else
                errors = errors + 1
                msg.error("Nil Found -> X:" .. tostring(cam_dx) .. " Y:" .. tostring(cam_dy) .. " Z:" .. tostring(cam_dz) .. " Rx:" .. tostring(rx) .. " Ry:" .. tostring(ry) .. " Rz:" .. tostring(rz))
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