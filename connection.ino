#define LIGHT_PIN 5  // You've connected LED to D5 (GPIO 5)

void setup() {
  Serial.begin(9600);
  pinMode(LIGHT_PIN, OUTPUT);
  digitalWrite(LIGHT_PIN, LOW); // LED starts OFF
}

void loop() {
  if (Serial.available()) {
    char input = Serial.read();
    if (input == '1') {
      digitalWrite(LIGHT_PIN, HIGH); // LED ON
      Serial.println("LED ON");
    } else if (input == '0') {
      digitalWrite(LIGHT_PIN, LOW);  // LED OFF
      Serial.println("LED OFF");
    }
  }
}
