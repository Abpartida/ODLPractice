package org.example;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.WebSocket;
import okhttp3.WebSocketListener;
import okio.ByteString;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Instant;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

/**
 * Minimal interactive console that sends rover control commands over a WebSocket using OkHttp.
 */
public final class App {
    private static final String DEFAULT_WS_URL = "ws://127.0.0.1:5000/ws/control";
    private static final Gson GSON = new GsonBuilder().disableHtmlEscaping().create();

    private App() {}

    public static void main(String[] args) throws IOException {
        String envUrl = System.getenv("CONTROL_WS_URL");
        String wsUrl = args.length > 0 ? args[0] : (envUrl == null || envUrl.isBlank() ? DEFAULT_WS_URL : envUrl);
        System.out.println("Using control WebSocket: " + wsUrl);
        try (ControlConsole console = new ControlConsole(wsUrl)) {
            console.start();
        }
    }

    private static final class ControlConsole implements AutoCloseable {
        private final OkHttpClient client;
        private final WebSocket socket;
        private final BufferedReader stdin;
        private final String wsUrl;

        ControlConsole(String wsUrl) {
            this.wsUrl = wsUrl;
            this.client = new OkHttpClient();
            Request request = new Request.Builder().url(wsUrl).build();
            this.socket = client.newWebSocket(request, new ControlSocketListener());
            this.stdin = new BufferedReader(new InputStreamReader(System.in));
        }

        void start() throws IOException {
            printHelp();
            String line;
            while ((line = stdin.readLine()) != null) {
                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }
                if ("quit".equalsIgnoreCase(trimmed) || "exit".equalsIgnoreCase(trimmed)) {
                    System.out.println("Disconnecting...");
                    break;
                }
                if ("help".equalsIgnoreCase(trimmed)) {
                    printHelp();
                    continue;
                }
                try {
                    String payload = CommandEncoder.encode(trimmed);
                    if (payload == null) {
                        System.out.println("Unrecognized command. Type 'help' for options.");
                        continue;
                    }
                    boolean accepted = socket.send(payload);
                    if (!accepted) {
                        System.out.println("WebSocket rejected the command; closing.");
                        break;
                    }
                } catch (IllegalArgumentException ex) {
                    System.out.println("⚠️  " + ex.getMessage());
                }
            }
        }

        private void printHelp() {
            System.out.println("""

Available commands:
  drive forward|backward|left|right|stop
  drive axes <x> <y>         (floating point values -1.0..1.0)
  lift up|down|stop
  fan on|off
  mode get | mode set <manual|autonomous>
  status                     (probe serial + mode)
  ping                       (keep-alive)
  help                       (show this list)
  quit / exit                (close console)
""");
        }

        @Override
        public void close() {
            try {
                stdin.close();
            } catch (IOException ignored) {
            }
            socket.close(1000, "console shutdown");
            client.dispatcher().executorService().shutdown();
            client.connectionPool().evictAll();
            System.out.println("Console closed.");
        }
    }

    private static final class ControlSocketListener extends WebSocketListener {
        @Override
        public void onOpen(WebSocket webSocket, okhttp3.Response response) {
            System.out.println("[WS] Connected (" + response.code() + ").");
        }

        @Override
        public void onMessage(WebSocket webSocket, String text) {
            System.out.println("[WS] " + text);
        }

        @Override
        public void onMessage(WebSocket webSocket, ByteString bytes) {
            System.out.println("[WS] " + bytes.utf8());
        }

        @Override
        public void onClosing(WebSocket webSocket, int code, String reason) {
            System.out.println("[WS] Closing (" + code + "): " + reason);
        }

        @Override
        public void onFailure(WebSocket webSocket, Throwable t, okhttp3.Response response) {
            String code = response == null ? "n/a" : Integer.toString(response.code());
            System.out.println("[WS] Failure (code=" + code + "): " + t.getMessage());
        }
    }

    private static final class CommandEncoder {
        private CommandEncoder() {}

        static String encode(String input) {
            String[] tokens = input.trim().split("\\s+");
            if (tokens.length == 0) {
                return null;
            }
            String root = tokens[0].toLowerCase(Locale.US);
            return switch (root) {
                case "drive" -> encodeDrive(tokens);
                case "lift" -> encodeSimple("lift", tokens, 2);
                case "fan" -> encodeSimple("fan", tokens, 2);
                case "mode" -> encodeMode(tokens);
                case "status" -> baseMessage("status");
                case "ping" -> baseMessage("ping");
                default -> null;
            };
        }

        private static String encodeDrive(String[] tokens) {
            if (tokens.length < 2) {
                throw new IllegalArgumentException("Drive command requires a direction or 'axes <x> <y>'.");
            }
            if ("axes".equalsIgnoreCase(tokens[1])) {
                if (tokens.length < 4) {
                    throw new IllegalArgumentException("Drive axes command requires both x and y values.");
                }
                double x = parseDouble(tokens[2], "x");
                double y = parseDouble(tokens[3], "y");
                Map<String, Object> msg = baseMessageAsMap("drive");
                msg.put("x", x);
                msg.put("y", y);
                return GSON.toJson(msg);
            }
            Map<String, Object> msg = baseMessageAsMap("drive");
            msg.put("command", tokens[1].toUpperCase(Locale.US));
            return GSON.toJson(msg);
        }

        private static String encodeSimple(String type, String[] tokens, int required) {
            if (tokens.length < required) {
                throw new IllegalArgumentException(type + " command requires an argument.");
            }
            Map<String, Object> msg = baseMessageAsMap(type);
            msg.put("command", tokens[1].toLowerCase(Locale.US));
            return GSON.toJson(msg);
        }

        private static String encodeMode(String[] tokens) {
            Map<String, Object> msg = baseMessageAsMap("mode");
            if (tokens.length == 1 || "get".equalsIgnoreCase(tokens[1])) {
                msg.put("command", "get");
                return GSON.toJson(msg);
            }
            if ("set".equalsIgnoreCase(tokens[1])) {
                if (tokens.length < 3) {
                    throw new IllegalArgumentException("Mode set command requires 'manual' or 'autonomous'.");
                }
                msg.put("command", "set");
                msg.put("value", tokens[2].toLowerCase(Locale.US));
                return GSON.toJson(msg);
            }
            throw new IllegalArgumentException("Unsupported mode verb: " + tokens[1]);
        }

        private static double parseDouble(String token, String name) {
            try {
                return Double.parseDouble(token);
            } catch (NumberFormatException ex) {
                throw new IllegalArgumentException("Invalid " + name + " value: " + token);
            }
        }

        private static String baseMessage(String type) {
            return GSON.toJson(baseMessageAsMap(type));
        }

        private static Map<String, Object> baseMessageAsMap(String type) {
            Map<String, Object> msg = new LinkedHashMap<>();
            msg.put("id", UUID.randomUUID().toString());
            msg.put("type", type);
            msg.put("client_ts", Instant.now().toString());
            return msg;
        }
    }
}
