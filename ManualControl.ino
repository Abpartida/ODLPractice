/*
  Arduino Nano ESP32 (ESP32-S3) - 4x DC Motor Control via Sabertooth 2-pin input
  + USB Serial command interface from Raspberry Pi

  You confirmed:
  - Sabertooth drivers are configured for 2-pin input per channel (M1A/M1B and M2A/M2B)
  - You are using A0-A7 as those 8 control pins.

  Pin mapping (edit if your wiring differs):
    Motor1: A0=M1A, A1=M1B
    Motor2: A2=M2A, A3=M2B
    Motor3: A4=M3A, A5=M3B
    Motor4: A6=M4A, A7=M4B

  Control law (signed command -255..255):
    >0 : PWM on "A" pin, "B" pin LOW
    <0 : PWM on "B" pin, "A" pin LOW
    =0 : both LOW (coast/stop)

  Serial protocol (newline-terminated):
    PING
    STOP
    STAT?
    W <m1> <m2> <m3> <m4>     (each -255..255)
    M <idx> <val>              (idx 1..4, val -255..255)
    RAMP <0|1>                 (disable/enable ramp)
    FAILSAFE <ms>              (set failsafe window, e.g. 1200)

  Responses:
    OK
    ERR <reason>
    STAT m1=.. m2=.. m3=.. m4=.. ramp=.. failsafe_ms=.. age_ms=..

  Safety:
    - Failsafe: if no valid command arrives for FAILSAFE_MS -> STOP ALL
*/

#include <Arduino.h>

// ===================== CONFIG (defaults) =====================
static const uint32_t SERIAL_BAUD_DEFAULT = 115200;

// PWM
static const uint32_t PWM_HZ = 20000; // 20kHz quiet PWM
static const uint8_t PWM_BITS = 8;    // 0..255

// Ramp (smooth acceleration)
static bool g_enableRamp = true;
static const uint32_t RAMP_INTERVAL_MS = 20;
static const int RAMP_STEP = 12; // max change per interval in -255..255 units

// Failsafe
static uint32_t g_failsafeMs = 1200;

// ===================== PIN MAP (A0-A7) =====================
static const int M1A = A0;
static const int M1B = A1;
static const int M2A = A2;
static const int M2B = A3;
static const int M3A = A4;
static const int M3B = A5;
static const int M4A = A6;
static const int M4B = A7;

// ===================== INTERNALS =====================
struct Motor2Pin
{
    int a;
    int b;
    int target;  // -255..255
    int applied; // -255..255
};

static Motor2Pin motors[4] = {
    {M1A, M1B, 0, 0},
    {M2A, M2B, 0, 0},
    {M3A, M3B, 0, 0},
    {M4A, M4B, 0, 0},
};

static uint32_t lastValidCmdMs = 0;
static uint32_t lastRampMs = 0;

static String lineBuf;

// ===================== HELPERS =====================
static int clamp255(int v)
{
    if (v > 255)
        return 255;
    if (v < -255)
        return -255;
    return v;
}

static int rampToward(int current, int target, int step)
{
    if (current < target)
    {
        current += step;
        if (current > target)
            current = target;
    }
    else if (current > target)
    {
        current -= step;
        if (current < target)
            current = target;
    }
    return current;
}

static void motorWrite2Pin(const Motor2Pin &m, int cmd /*-255..255*/)
{
    cmd = clamp255(cmd);
    int mag = abs(cmd);

    if (mag == 0)
    {
        // Coast/stop
        analogWrite(m.a, 0);
        analogWrite(m.b, 0);
        digitalWrite(m.a, LOW);
        digitalWrite(m.b, LOW);
        return;
    }

    if (cmd > 0)
    {
        // Forward: PWM on A, B forced low
        analogWrite(m.b, 0);
        digitalWrite(m.b, LOW);
        analogWrite(m.a, mag);
    }
    else
    {
        // Reverse: PWM on B, A forced low
        analogWrite(m.a, 0);
        digitalWrite(m.a, LOW);
        analogWrite(m.b, mag);
    }
}

static void stopAll()
{
    for (int i = 0; i < 4; i++)
    {
        motors[i].target = 0;
        motors[i].applied = 0;
        motorWrite2Pin(motors[i], 0);
    }
}

static bool parseIntToken(const String &tok, int &out)
{
    if (tok.length() == 0)
        return false;
    char *endptr = nullptr;
    long v = strtol(tok.c_str(), &endptr, 10);
    if (endptr == tok.c_str() || *endptr != '\0')
        return false;
    out = (int)v;
    return true;
}

static void replyStat()
{
    uint32_t age = millis() - lastValidCmdMs;
    Serial.printf(
        "STAT m1=%d m2=%d m3=%d m4=%d ramp=%d failsafe_ms=%lu age_ms=%lu\n",
        motors[0].applied, motors[1].applied, motors[2].applied, motors[3].applied,
        g_enableRamp ? 1 : 0,
        (unsigned long)g_failsafeMs,
        (unsigned long)age);
}

static void markValidCmd()
{
    lastValidCmdMs = millis();
}

// ===================== COMMAND HANDLER =====================
static void handleLine(String line)
{
    line.trim();
    if (line.length() == 0)
        return;

    const int MAXTOK = 10;
    String tok[MAXTOK];
    int n = 0;

    int start = 0;
    while (start < (int)line.length() && n < MAXTOK)
    {
        while (start < (int)line.length() && line[start] == ' ')
            start++;
        if (start >= (int)line.length())
            break;
        int end = line.indexOf(' ', start);
        if (end < 0)
            end = line.length();
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
            motors[i].target = clamp255(v);
            if (!g_enableRamp)
            {
                motors[i].applied = motors[i].target;
                motorWrite2Pin(motors[i], motors[i].applied);
            }
        }
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
        motors[idx - 1].target = clamp255(val);
        if (!g_enableRamp)
        {
            motors[idx - 1].applied = motors[idx - 1].target;
            motorWrite2Pin(motors[idx - 1], motors[idx - 1].applied);
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
    Serial.begin(SERIAL_BAUD_DEFAULT);
    delay(200);

    // Configure PWM globally for Arduino-ESP32 core
    analogWriteFrequency(PWM_HZ);
    analogWriteResolution(PWM_BITS);

    // Configure pins
    for (int i = 0; i < 4; i++)
    {
        pinMode(motors[i].a, OUTPUT);
        pinMode(motors[i].b, OUTPUT);
        digitalWrite(motors[i].a, LOW);
        digitalWrite(motors[i].b, LOW);
        analogWrite(motors[i].a, 0);
        analogWrite(motors[i].b, 0);
    }

    stopAll();
    lastValidCmdMs = millis();
    lastRampMs = millis();

    Serial.println("OK boot");
}

void loop()
{
    // Read serial lines
    while (Serial.available() > 0)
    {
        char c = (char)Serial.read();
        if (c == '\r')
            continue;
        if (c == '\n')
        {
            handleLine(lineBuf);
            lineBuf = "";
        }
        else
        {
            if (lineBuf.length() < 220)
                lineBuf += c;
        }
    }

    uint32_t now = millis();

    // Failsafe
    if (now - lastValidCmdMs > g_failsafeMs)
    {
        stopAll();
        lastValidCmdMs = now; // prevents repeated hammering if Pi is offline
    }

    // Ramping update
    if (g_enableRamp && (now - lastRampMs >= RAMP_INTERVAL_MS))
    {
        lastRampMs = now;
        for (int i = 0; i < 4; i++)
        {
            int next = rampToward(motors[i].applied, motors[i].target, RAMP_STEP);
            if (next != motors[i].applied)
            {
                motors[i].applied = next;
                motorWrite2Pin(motors[i], motors[i].applied);
            }
        }
    }
}