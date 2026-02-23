import serial, time

PORT = "/dev/ttys004"   # <-- the OTHER end of the socat pair
BAUD = 115200

state = {"m": [0,0,0,0], "v": 0}

def handle(line: str):
    line = line.strip()
    if not line:
        return None
    parts = line.split()
    cmd = parts[0].upper()

    if cmd == "PING":
        return "OK\n"
    if cmd == "STOP":
        state["m"] = [0,0,0,0]
        state["v"] = 0
        return "OK\n"
    if cmd == "STAT?":
        m1,m2,m3,m4 = state["m"]
        return f"STAT m1={m1} m2={m2} m3={m3} m4={m4} age_ms=0\n"
    if cmd == "W" and len(parts) == 5:
        vals = [int(x) for x in parts[1:5]]
        state["m"] = vals
        return "OK\n"
    if cmd == "V" and len(parts) == 2:
        state["v"] = int(parts[1])
        return "OK\n"

    return "ERR unknown\n"

ser = serial.Serial(PORT, BAUD, timeout=0.1)
print("ESP32 emulator on", PORT)

last_stat = 0.0
while True:
    line = ser.readline().decode(errors="ignore")
    if line:
        print("ESP32 RECEIVED:", line)
        resp = handle(line)
        if resp:
            ser.write(resp.encode())

    # Optional periodic telemetry
    now = time.time()
    if now - last_stat > 0.2:
        m1,m2,m3,m4 = state["m"]
        ser.write(f"STAT m1={m1} m2={m2} m3={m3} m4={m4} age_ms=0\n".encode())
        last_stat = now