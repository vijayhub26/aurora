#define LIGHT_PIN 5  // Onboard LED for most ESP32 boards

void setup() {
  Serial.begin(9600);            // Start serial
  pinMode(LIGHT_PIN, OUTPUT);    // Set LED pin as output
  digitalWrite(LIGHT_PIN, LOW);  // Make sure light is OFF initially
}

void loop() {
  if (Serial.available()) {
    char input = Serial.read();  // Read one byte from serial
    
    if (input == '1') {
      digitalWrite(LIGHT_PIN, HIGH);  // LED ON
      Serial.println("LED ON");
    } else if (input == '0') {
      digitalWrite(LIGHT_PIN, LOW);   // LED OFF
      Serial.println("LED OFF");
    }
  }
}
