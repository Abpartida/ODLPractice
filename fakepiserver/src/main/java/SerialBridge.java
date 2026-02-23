import com.fazecast.jSerialComm.SerialPort;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class SerialBridge {

    private final SerialPort serialPort;
    private final InputStream inputStream;
    private final OutputStream outputStream;

    public SerialBridge(String portName, int baudRate) {
        serialPort = SerialPort.getCommPort(portName);
        serialPort.setBaudRate(baudRate);
        serialPort.setNumDataBits(8);
        serialPort.setNumStopBits(1);
        serialPort.setParity(SerialPort.NO_PARITY);
        serialPort.setComPortTimeouts(
                SerialPort.TIMEOUT_READ_SEMI_BLOCKING,
                200,   // read timeout ms
                0
        );

        if (!serialPort.openPort()) {
            throw new RuntimeException("Failed to open serial port: " + portName);
        }

        inputStream = serialPort.getInputStream();
        outputStream = serialPort.getOutputStream();

        System.out.println("SerialBridge connected to " + portName);
    }

    public synchronized String send(String command) throws IOException {
        String fullCommand = command + "\n";
        outputStream.write(fullCommand.getBytes());
        outputStream.flush();

        StringBuilder response = new StringBuilder();
        long start = System.currentTimeMillis();

        while (System.currentTimeMillis() - start < 300) {
            if (inputStream.available() > 0) {
                int b = inputStream.read();
                if (b == -1) break;
                response.append((char) b);
                if (b == '\n') break;
            }
        }

        return response.toString();
    }

    public synchronized void close() {
        try {
            inputStream.close();
            outputStream.close();
        } catch (IOException ignored) {}

        serialPort.closePort();
        System.out.println("SerialBridge closed.");
    }
}
