#include <SabertoothSimplified.h>

// Hardware Serial0 interface for Sabertooth primary motor controller
 SabertoothSimplified ST(Serial0); 

// Pin definitions for auxiliary systems 
const int Fan1 = D2; // Relay control for Fan 1
const int Fan2 = D3; // Relay control for Fan 2
const int RPWM_PIN = D4;  // BTS7960 Right PWM 
const int LPWM_PIN = D5;  // BTS7960 Left PWM 
const int LIDAR_RX = D6; 
const int LIDAR_TX = D7;

// System constraints and watchdog timing parameters
unsigned long lastCommandTime = 0;
unsigned long lastLidarTime = 0;
const unsigned long WATCHDOG_TIMEOUT = 10000; // Communication timeout threshold in milliseconds

// Kinematic constraints (Amplitude range: 0-127)
const int DRIVE_SPEED = 127; 
const int TURN_SPEED = 127;  
const byte request[] = {0x01, 0x03, 0x00, 0x00, 0x00, 0x01, 0x84, 0x0A}; 
float currentDist = 0;

void setup() {
  Serial0.begin(9600);    // Sabertooth UART baud rate
  Serial.begin(115200);   // Host system UART baud rate
  setupLidar();
  
  // Configure relay pins for fan control
  pinMode(Fan1, OUTPUT);
  pinMode(Fan2, OUTPUT);
  
  // Configure BTS7960 H-bridge control pins
  pinMode(RPWM_PIN, OUTPUT);
  pinMode(LPWM_PIN, OUTPUT);
  
  // Initialize auxiliary actuator to a disabled state
  digitalWrite(RPWM_PIN, LOW);
  digitalWrite(LPWM_PIN, LOW);

  digitalWrite(Fan1, HIGH); 
  digitalWrite(Fan2, HIGH);

  delay(2000); // Hardware initialization allowance
  
  // Initialize primary drive system to a stationary state
  ST.motor(1, 0);
  ST.motor(2, 0);
}

void setupLidar() {
  // Configuration: 115200 baud, 8 data bits, No parity, 1 stop bit
  Serial1.begin(115200, SERIAL_8N1, LIDAR_RX, LIDAR_TX); 
}

void readLidar() {
  // Flush incoming serial buffer to clear residual data
  while (Serial1.available()) {
    Serial1.read();
  }

  // Transmit Modbus RTU request payload
  Serial1.write(request, sizeof(request));

  // Processing delay for sensor response propagation
  delay(50); 

  // Validate received frame length
  if (Serial1.available() >= 7) {
    byte response[7];
    
    // Buffer incoming payload data
    for (int i = 0; i < 7; i++) {
      response[i] = Serial1.read();
    }

    // Verify Modbus address and function code integrity
    if (response[0] == 0x01 && response[1] == 0x03) {
      
      // Reconstruct 16-bit integer from High and Low data bytes
      int rawCm = (response[3] << 8) | response[4];

      // Apply unit conversion factor from centimeters to feet
      currentDist = rawCm / 30.48; 

      // Transmit formatted distance payload to host system
      Serial.print("LIDAR Distance: ");
      
      // Format floating-point output to two decimal precision
      Serial.println(currentDist, 2); 
    }
  }
}

void loop() {
  // Parse incoming serial commands from host
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    // Reset watchdog timeout counter
    lastCommandTime = millis(); 

    // Execute system functions based on host commands
    if (command == "FORWARD") {
      ST.motor(1, DRIVE_SPEED);
      ST.motor(2, DRIVE_SPEED);
      digitalWrite(Fan1, LOW); 
      digitalWrite(Fan2, LOW); 
    }
    else if (command == "BACKWARD") {
      ST.motor(1, -DRIVE_SPEED); 
      ST.motor(2, -DRIVE_SPEED);
      digitalWrite(Fan1, LOW); 
      digitalWrite(Fan2, LOW); 
    } 
    else if (command == "LEFT") {
      ST.motor(1, -TURN_SPEED); 
      ST.motor(2, TURN_SPEED); 
      digitalWrite(Fan1, LOW); 
      digitalWrite(Fan2, LOW);  
    } 
    else if (command == "RIGHT") {
      ST.motor(1, TURN_SPEED);  
      ST.motor(2, -TURN_SPEED); 
      digitalWrite(Fan1, LOW); 
      digitalWrite(Fan2, LOW); 
    } 
    
    // Auxiliary H-bridge actuator commands
    else if (command == "UP") {
      // Actuate auxiliary mechanism (Forward polarity)
      digitalWrite(LPWM_PIN, LOW);   
      digitalWrite(RPWM_PIN, HIGH); 
      digitalWrite(Fan1, HIGH); 
      digitalWrite(Fan2, HIGH); 
    }
    else if (command == "DOWN") {
      // Actuate auxiliary mechanism (Reverse polarity)
      digitalWrite(RPWM_PIN, LOW);   
      digitalWrite(LPWM_PIN, HIGH);
      digitalWrite(Fan1, HIGH); 
      digitalWrite(Fan2, HIGH); 
    }
    else if (command == "FAN"){
      digitalWrite(Fan1, LOW); 
      digitalWrite(Fan2, LOW); 
    }
    else if (command == "STOP") {
      // Disengage primary drive system
      ST.motor(1, 0);
      ST.motor(2, 0);
      
      // Apply dynamic braking to auxiliary actuator
      digitalWrite(RPWM_PIN, LOW);
      digitalWrite(LPWM_PIN, LOW);
      
      digitalWrite(Fan1, HIGH); 
      digitalWrite(Fan2, HIGH); 
    }
  }

  // Watchdog handler: Disengage all actuators upon loss of host communication
  if (millis() - lastCommandTime > WATCHDOG_TIMEOUT) {
    ST.motor(1, 0);
    ST.motor(2, 0);
    digitalWrite(RPWM_PIN, LOW);
    digitalWrite(LPWM_PIN, LOW);
    digitalWrite(Fan1, HIGH); 
    digitalWrite(Fan2, HIGH);
  }
  if (millis() - lastLidarTime > 100) {
    lastLidarTime = millis();
    readLidar(); 
  }
}
