#define LED_BUILTIN 2  
#define BUTTON_PIN 0  
bool blinkstate = false;
void setup() {
  pinMode(LED_BUILTIN, OUTPUT); 
  pinMode(BUTTON_PIN,INPUT_PULLUP);
}

void loop() {


static bool lastbuttonstate =HIGH;
 bool buttonstate = digitalRead(BUTTON_PIN);

if(buttonstate==LOW&& lastbuttonstate==HIGH){
  
  blinkstate=!blinkstate;
  delay(300);
  
  }

lastbuttonstate=buttonstate;


  if(blinkstate){
  digitalWrite(LED_BUILTIN, HIGH); 
  delay(1000);                     
  digitalWrite(LED_BUILTIN, LOW);  
  delay(1000);                     
}
}
