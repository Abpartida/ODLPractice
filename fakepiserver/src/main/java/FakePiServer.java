import okhttp3.mockwebserver.Dispatcher;
import okhttp3.mockwebserver.MockResponse;
import okhttp3.mockwebserver.MockWebServer;
import okhttp3.mockwebserver.RecordedRequest;

import java.net.InetAddress;

public class FakePiServer {

    // ===== CONFIG =====
    // Set this to the FIRST PTY printed by socat (the "Pi side")
    private static final String SERIAL_PORT_NAME = "/dev/ttys003";
    private static final int SERIAL_BAUD = 115200;

    // Fake Pi HTTP server port
    private static final int HTTP_PORT = 8080;

    private static SerialBridge serial;

    public static void main(String[] args) throws Exception {
        System.out.println("Starting FakePiServer...");

        // Open serial once for the lifetime of the server
        serial = new SerialBridge(SERIAL_PORT_NAME, SERIAL_BAUD);

        MockWebServer server = new MockWebServer();
        server.setDispatcher(new ApiDispatcher());

        server.start(InetAddress.getByName("0.0.0.0"), HTTP_PORT);

        System.out.println("FakePiServer running on http://localhost:" + HTTP_PORT);
        System.out.println("Serial connected to: " + SERIAL_PORT_NAME + " @ " + SERIAL_BAUD);
        System.out.println("Endpoints:");
        System.out.println("  POST /api/drive/joystick  body: {\"x\":0.0,\"y\":0.8}");
        System.out.println("  POST /api/drive/stop");
        System.out.println("  POST /auth/login");
        System.out.println("  GET  /status");

        // Clean shutdown
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                server.shutdown();
            } catch (Exception ignored) {}
            try {
                if (serial != null) serial.close();
            } catch (Exception ignored) {}
        }));
    }

    /**
     * Dispatcher that mimics a small subset of your FastAPI endpoints.
     *
     * /drive mixes joystick (x,y) -> left/right -> W command:
     *   left  = clamp(y + x)
     *   right = clamp(y - x)
     *   then scaled to -255..255
     *
     * Sends: W left left right right\n
     */
    private static class ApiDispatcher extends Dispatcher {
        @Override
        public MockResponse dispatch(RecordedRequest request) {
            try {
                String path = request.getPath();
                String method = request.getMethod();

                if (path == null || method == null) {
                    return json(400, "{\"error\":\"bad request\"}");
                }

                // --- POST /auth/login ---
                // App expects this during startup sometimes. Return a minimal valid body.
                if ((path.equals("/auth/login") || path.equals("auth/login")) && method.equalsIgnoreCase("POST")) {
                    return json(200, "{\"access_token\":\"mock-token\",\"token_type\":\"bearer\",\"expires_in\":3600,\"message\":\"Mock login ok\"}");
                }

                // --- POST /api/drive/joystick ---
                // This matches your Retrofit interface: @POST("/api/drive/joystick")
                if (path.equals("/api/drive/joystick") && method.equalsIgnoreCase("POST")) {
                    String body = request.getBody().readUtf8();
                    System.out.println("HTTP /api/drive/joystick body: " + body);

                    // Support common joystick payloads. Primary: {"x":...,"y":...}
                    double x = extractDouble(body, "x");
                    double y = extractDouble(body, "y");

                    // If your DTO uses different field names, we can fall back to these:
                    // e.g. {"horizontal":...,"vertical":...}
                    if (x == 0.0 && y == 0.0) {
                        x = extractDouble(body, "horizontal");
                        y = extractDouble(body, "vertical");
                    }

                    // Optional digital directions (e.g. {"direction":"Forward"})
                    String dir = extractString(body, "direction");
                    if ((x == 0.0 && y == 0.0) && dir != null && !dir.isEmpty()) {
                        double throttle = 0.8;
                        switch (dir) {
                            case "Forward":
                                x = 0.0; y = throttle;
                                break;
                            case "Reverse":
                                x = 0.0; y = -throttle;
                                break;
                            case "Left":
                                x = -throttle; y = 0.0;
                                break;
                            case "Right":
                                x = throttle; y = 0.0;
                                break;
                            case "Forward-Left":
                                x = -throttle * 0.6; y = throttle;
                                break;
                            case "Forward-Right":
                                x = throttle * 0.6; y = throttle;
                                break;
                            case "Reverse-Left":
                                x = -throttle * 0.6; y = -throttle;
                                break;
                            case "Reverse-Right":
                                x = throttle * 0.6; y = -throttle;
                                break;
                            default:
                                break;
                        }
                    }

                    int left = (int) Math.round(clamp(y + x) * 255.0);
                    int right = (int) Math.round(clamp(y - x) * 255.0);

                    String serialCmd = String.format("W %d %d %d %d", left, left, right, right);
                    System.out.println("Serial TX: " + serialCmd + " (dir=" + (dir == null ? "" : dir) + ", x=" + x + ", y=" + y + ")");

                    String serialResp = serial.send(serialCmd);
                    String sanitized = serialResp == null ? "" : serialResp.trim();
                    System.out.println("Serial RX: " + sanitized);

                    return json(200, "{\"status\":\"ok\",\"cmd\":\"" + escape(serialCmd) + "\",\"serial\":\"" + escape(sanitized) + "\"}");
                }

                // --- POST /api/drive/stop ---
                if (path.equals("/api/drive/stop") && method.equalsIgnoreCase("POST")) {
                    String serialResp = serial.send("STOP");
                    String sanitized = serialResp == null ? "" : serialResp.trim();
                    return json(200, "{\"status\":\"stopped\",\"serial\":\"" + escape(sanitized) + "\"}");
                }

                // --- GET /status ---
                if (path.equals("/status") && method.equalsIgnoreCase("GET")) {
                    // Ask the emulator/ESP32 for status (optional)
                    String serialResp = serial.send("STAT?");
                    String sanitized = serialResp == null ? "" : serialResp.trim();
                    return json(200, "{\"system_status\":\"Nominal\",\"serial\":\"" + escape(sanitized) + "\"}");
                }

                return json(404, "{\"error\":\"not found\"}");

            } catch (Exception e) {
                e.printStackTrace();
                return json(500, "{\"error\":\"" + escape(String.valueOf(e.getMessage())) + "\"}");
            }
        }

        private MockResponse json(int code, String body) {
            return new MockResponse()
                    .setResponseCode(code)
                    .addHeader("Content-Type", "application/json")
                    .setBody(body);
        }

        private double clamp(double v) {
            if (v > 1.0) return 1.0;
            if (v < -1.0) return -1.0;
            return v;
        }

        private String extractString(String json, String key) {
            try {
                int idx = json.indexOf("\"" + key + "\"");
                if (idx < 0) return null;
                int colon = json.indexOf(':', idx);
                if (colon < 0) return null;

                int i = colon + 1;
                while (i < json.length() && Character.isWhitespace(json.charAt(i))) i++;
                if (i >= json.length() || json.charAt(i) != '\"') return null;
                i++;

                StringBuilder sb = new StringBuilder();
                boolean escaping = false;
                for (; i < json.length(); i++) {
                    char c = json.charAt(i);
                    if (escaping) {
                        sb.append(c);
                        escaping = false;
                    } else if (c == '\\') {
                        escaping = true;
                    } else if (c == '\"') {
                        break;
                    } else {
                        sb.append(c);
                    }
                }
                return sb.toString();
            } catch (Exception ignored) {
                return null;
            }
        }

        // Minimal JSON value extractor for numbers: {"x":0.1,"y":-0.4}
        private double extractDouble(String json, String key) {
            try {
                int idx = json.indexOf("\"" + key + "\"");
                if (idx < 0) return 0.0;

                int colon = json.indexOf(':', idx);
                if (colon < 0) return 0.0;

                // Find end of number token
                int end = colon + 1;
                while (end < json.length() && Character.isWhitespace(json.charAt(end))) end++;

                int start = end;
                while (end < json.length()) {
                    char c = json.charAt(end);
                    if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
                        end++;
                    } else {
                        break;
                    }
                }

                String val = json.substring(start, end).trim();
                if (val.isEmpty()) return 0.0;
                return Double.parseDouble(val);

            } catch (Exception ignored) {
                return 0.0;
            }
        }

        // Escape quotes/backslashes for safe JSON string embedding
        private String escape(String s) {
            if (s == null) return "";
            return s.replace("\\", "\\\\").replace("\"", "\\\"");
        }
    }
}
