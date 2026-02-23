/*
  Arduino Nano ESP32 (ESP32-S3) - 4x DC Motor Control
  TWO Sabertooth motor drivers using Simplified Serial (UART)
  + USB Serial command interface from Raspberry Pi

  Data flow:
    Raspberry Pi (USB Serial @ 115200) -> Nano ESP32 (this firmware)
    Nano ESP32 Serial0 TX @ 9600 -> Sabertooth #1 S1 (motors 1 & 2)
    Nano ESP32 Serial1 TX @ 9600 -> Sabertooth #2 S1 (motors 3 & 4)

  Wiring notes:
    - TX only is needed for Simplified Serial (to Sabertooth S1)
    - All grounds MUST be shared (Pi / ESP32 / Sabertooths)
    - Set Sabertooth DIP switches for Simplified Serial mode per your model manual

  Sabertooth command range:
    -127..127 (signed)

  Serial protocol (newline-terminated) over USB Serial (from Pi):
    PING
    STOP
    STAT?

    // Low-level direct motor control from Pi:
    // Accepts values in -255..255 (common app-side), firmware scales to -127..127.
    W <m1> <m2> <m3> <m4>
    M <idx> <val>          (idx 1..4)

    // Higher-level (matches Android direction strings):
    DRIVE <Direction>      (Forward, Reverse, Left, Right,
                            Forward-Left, Forward-Right, Reverse-Left, Reverse-Right)
    SPEED <0..255>         (default throttle used by DRIVE; scaled to 0..127 internally)

    // Stubs (so Pi can forward endpoints even before wiring those GPIOs):
    LIFT <UP|DOWN|STOP>
    FAN <ON|OFF>

    // Tuning:
    RAMP <0|1>             (disable/enable ramp)
    FAILSAFE <ms>          (100..60000)

  Responses:
    OK
    ERR <reason>
    STAT m1=.. m2=.. m3=.. m4=.. ramp=.. speed=.. failsafe_ms=.. age_ms=..

  Safety:
    - Failsafe: if no valid command arrives within FAILSAFE -> STOP ALL
*/

#include <Arduino.h>
#include <math.h>
#include <SabertoothSimplified.h>

// ===================== CONFIG =====================
static const uint32_t USB_BAUD = 115200;
static const uint32_t SABER_BAUD = 9600;

// Ramp (smooth acceleration)
static bool g_enableRamp = true;
static const uint32_t RAMP_INTERVAL_MS = 20;
static const int RAMP_STEP = 12; // max change per interval in -127..127 units

// DRIVE speed in Sabertooth units (0..127)
static int g_driveSpeed = 100;

// Failsafe window (ms)
static uint32_t g_failsafeMs = 1200;

// ===================== STATE =====================
static int g_targetM[4] = {0, 0, 0, 0};   // -127..127
static int g_appliedM[4] = {0, 0, 0, 0};  // -127..127

static uint32_t g_lastValidCmdMs = 0;
static uint32_t g_lastRampMs = 0;

static String g_lineBuf;

// ===================== SABERTOOTH =====================
// Sabertooth #1 on Serial0 controls motors 1 & 2
SabertoothSimplified sab1(Serial0);
// Sabertooth #2 on Serial1 controls motors 3 & 4
SabertoothSimplified sab2(Serial1);

// ===================== HELPERS =====================
static int clamp127(int v)
{
  if (v > 127) return 127;
  if (v < -127) return -127;
  return v;
}

static int clamp127Unsigned(int v)
{
  if (v > 127) return 127;
  if (v < 0) return 0;
  return v;
}

static int rampToward(int current, int target, int step)
{
  if (current < target)
  {
    current += step;
    if (current > target) current = target;
  }
  else if (current > target)
  {
    current -= step;
    if (current < target) current = target;
  }
  return current;
}

static bool parseIntToken(const String &tok, int &out)
{
  if (tok.length() == 0) return false;
  char *endptr = nullptr;
  long v = strtol(tok.c_str(), &endptr, 10);
  if (endptr == tok.c_str() || *endptr != '\0') return false;
  out = (int)v;
  return true;
}

static void markValidCmd()
{
  g_lastValidCmdMs = millis();
}

// Convert -255..255 to -127..127 (rounded) for Sabertooth
static int scale255To127(int v255)
{
  if (v255 > 255) v255 = 255;
  if (v255 < -255) v255 = -255;
  int v127 = (int)lroundf((float)v255 / 2.0f);
  return clamp127(v127);
}

static void applyMotorCommands()
{
  // Ensure range before writing
  for (int i = 0; i < 4; i++) g_appliedM[i] = clamp127(g_appliedM[i]);

  // Sabertooth #1
  sab1.motor(1, g_appliedM[0]);
  sab1.motor(2, g_appliedM[1]);

  // Sabertooth #2
  sab2.motor(1, g_appliedM[2]);
  sab2.motor(2, g_appliedM[3]);
}

static void stopAll()
{
  for (int i = 0; i < 4; i++)
  {
    g_targetM[i] = 0;
    g_appliedM[i] = 0;
  }
  applyMotorCommands();
}

static void replyStat()
{
  uint32_t age = millis() - g_lastValidCmdMs;
  Serial.printf(
      "STAT m1=%d m2=%d m3=%d m4=%d ramp=%d speed=%d failsafe_ms=%lu age_ms=%lu\n",
      g_appliedM[0], g_appliedM[1], g_appliedM[2], g_appliedM[3],
      g_enableRamp ? 1 : 0,
      g_driveSpeed,
      (unsigned long)g_failsafeMs,
      (unsigned long)age);
}

// Map direction string into arcade drive (x,y) in -1..1
static bool directionToXY(const String &dir, float &x, float &y)
{
  if (dir == "Forward") { x = 0.0f; y = 1.0f; return true; }
  if (dir == "Reverse") { x = 0.0f; y = -1.0f; return true; }
  if (dir == "Left")    { x = -1.0f; y = 0.0f; return true; }
  if (dir == "Right")   { x = 1.0f; y = 0.0f; return true; }

  if (dir == "Forward-Left")  { x = -0.6f; y = 1.0f; return true; }
  if (dir == "Forward-Right") { x = 0.6f;  y = 1.0f; return true; }
  if (dir == "Reverse-Left")  { x = -0.6f; y = -1.0f; return true; }
  if (dir == "Reverse-Right") { x = 0.6f;  y = -1.0f; return true; }

  return false;
}

// ===================== COMMAND HANDLER =====================
static void handleLine(String line)
{
  line.trim();
  if (line.length() == 0) return;

  const int MAXTOK = 10;
  String tok[MAXTOK];
  int n = 0;

  int start = 0;
  while (start < (int)line.length() && n < MAXTOK)
  {
    while (start < (int)line.length() && line[start] == ' ') start++;
    if (start >= (int)line.length()) break;
    int end = line.indexOf(' ', start);
    if (end < 0) end = line.length();
    tok[n++] = line.substring(start, end);
    start = end + 1;
  }

  String cmd = tok[0];
  cmd.toUpperCase();

  if (cmd == "PING")
  {
    markValidCmd();
    Serial.println("OK");
    return;
  }

  if (cmd == "STOP")
  {
    markValidCmd();
    stopAll();
    Serial.println("OK");
    return;
  }

  if (cmd == "STAT?")
  {
    markValidCmd();
    replyStat();
    return;
  }

  if (cmd == "W")
  {
    if (n != 5)
    {
      Serial.println("ERR expected: W <m1> <m2> <m3> <m4>");
      return;
    }

    for (int i = 0; i < 4; i++)
    {
      int v;
      if (!parseIntToken(tok[i + 1], v))
      {
        Serial.println("ERR bad wheel value");
        return;
      }
      g_targetM[i] = scale255To127(v);
      if (!g_enableRamp)
      {
        g_appliedM[i] = g_targetM[i];
      }
    }

    if (!g_enableRamp)
      applyMotorCommands();

    markValidCmd();
    Serial.println("OK");
    return;
  }

  if (cmd == "M")
  {
    if (n != 3)
    {
      Serial.println("ERR expected: M <1..4> <val>");
      return;
    }

    int idx, val;
    if (!parseIntToken(tok[1], idx) || !parseIntToken(tok[2], val))
    {
      Serial.println("ERR bad M args");
      return;
    }

    if (idx < 1 || idx > 4)
    {
      Serial.println("ERR motor idx must be 1..4");
      return;
    }

    g_targetM[idx - 1] = scale255To127(val);
    if (!g_enableRamp)
    {
      g_appliedM[idx - 1] = g_targetM[idx - 1];
      applyMotorCommands();
    }

    markValidCmd();
    Serial.println("OK");
    return;
  }

  if (cmd == "SPEED")
  {
    if (n != 2)
    {
      Serial.println("ERR expected: SPEED <0..255>");
      return;
    }

    int v;
    if (!parseIntToken(tok[1], v))
    {
      Serial.println("ERR bad SPEED value");
      return;
    }

    // Accept 0..255 from app/Pi, convert to 0..127
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    g_driveSpeed = (int)lroundf((float)v / 2.0f);
    g_driveSpeed = clamp127Unsigned(g_driveSpeed);

    markValidCmd();
    Serial.println("OK");
    return;
  }

  if (cmd == "DRIVE")
  {
    if (n != 2)
    {
      Serial.println("ERR expected: DRIVE <Direction>");
      return;
    }

    String dir = tok[1]; // keep original case
    float x = 0.0f, y = 0.0f;
    if (!directionToXY(dir, x, y))
    {
      Serial.println("ERR unknown direction");
      return;
    }

    // Arcade mix: left = y + x, right = y - x
    float l = y + x;
    float r = y - x;
    if (l > 1.0f) l = 1.0f;
    if (l < -1.0f) l = -1.0f;
    if (r > 1.0f) r = 1.0f;
    if (r < -1.0f) r = -1.0f;

    int left = (int)lroundf(l * (float)g_driveSpeed);
    int right = (int)lroundf(r * (float)g_driveSpeed);

    // Tank mapping: m1,m2 = left; m3,m4 = right
    g_targetM[0] = clamp127(left);
    g_targetM[1] = clamp127(left);
    g_targetM[2] = clamp127(right);
    g_targetM[3] = clamp127(right);

    if (!g_enableRamp)
    {
      for (int i = 0; i < 4; i++) g_appliedM[i] = g_targetM[i];
      applyMotorCommands();
    }

    markValidCmd();
    Serial.println("OK");
    return;
  }

  // Stubs for endpoint passthrough
  if (cmd == "LIFT")
  {
    if (n != 2)
    {
      Serial.println("ERR expected: LIFT <UP|DOWN|STOP>");
      return;
    }
    String a = tok[1];
    a.toUpperCase();
    if (a != "UP" && a != "DOWN" && a != "STOP")
    {
      Serial.println("ERR bad LIFT arg");
      return;
    }
    markValidCmd();
    Serial.println("OK");
    return;
  }

  if (cmd == "FAN")
  {
    if (n != 2)
    {
      Serial.println("ERR expected: FAN <ON|OFF>");
      return;
    }
    String a = tok[1];
    a.toUpperCase();
    if (a != "ON" && a != "OFF")
    {
      Serial.println("ERR bad FAN arg");
      return;
    }
    markValidCmd();
    Serial.println("OK");
    return;
  }

  if (cmd == "RAMP")
  {
    if (n != 2)
    {
      Serial.println("ERR expected: RAMP <0|1>");
      return;
    }
    int v;
    if (!parseIntToken(tok[1], v))
    {
      Serial.println("ERR bad RAMP value");
      return;
    }
    g_enableRamp = (v != 0);
    markValidCmd();
    Serial.println("OK");
    return;
  }

  if (cmd == "FAILSAFE")
  {
    if (n != 2)
    {
      Serial.println("ERR expected: FAILSAFE <ms>");
      return;
    }

    int ms;
    if (!parseIntToken(tok[1], ms) || ms < 100 || ms > 60000)
    {
      Serial.println("ERR failsafe must be 100..60000 ms");
      return;
    }

    g_failsafeMs = (uint32_t)ms;
    markValidCmd();
    Serial.println("OK");
    return;
  }

  Serial.println("ERR unknown command");
}

// ===================== SETUP / LOOP =====================
void setup()
{
  // USB serial from Pi
  Serial.begin(USB_BAUD);

  // Sabertooth UARTs
  Serial0.begin(SABER_BAUD);
  Serial1.begin(SABER_BAUD);

  delay(2000); // allow Sabertooths to wake

  // Explicit STOP to ensure control
  sab1.motor(1, 0);
  sab1.motor(2, 0);
  sab2.motor(1, 0);
  sab2.motor(2, 0);

  stopAll();
  g_lastValidCmdMs = millis();
  g_lastRampMs = millis();

  Serial.println("System Ready. Waiting for Pi commands...");
}

void loop()
{
  // Read USB serial lines
  while (Serial.available() > 0)
  {
    char c = (char)Serial.read();
    if (c == '\r') continue;
    if (c == '\n')
    {
      handleLine(g_lineBuf);
      g_lineBuf = "";
    }
    else
    {
      if (g_lineBuf.length() < 220) g_lineBuf += c;
    }
  }

  uint32_t now = millis();

  // Failsafe
  if (now - g_lastValidCmdMs > g_failsafeMs)
  {
    stopAll();
    g_lastValidCmdMs = now; // prevents repeated hammering if Pi is offline
  }

  // Ramping update
  if (g_enableRamp && (now - g_lastRampMs >= RAMP_INTERVAL_MS))
  {
    g_lastRampMs = now;

    bool changed = false;
    for (int i = 0; i < 4; i++)
    {
      int next = rampToward(g_appliedM[i], g_targetM[i], RAMP_STEP);
      if (next != g_appliedM[i])
      {
        g_appliedM[i] = next;
        changed = true;
      }
    }

    if (changed)
    {
      applyMotorCommands();
    }
  }
}