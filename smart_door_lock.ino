/*
 * ============================================================
 *  Smart Door Lock  --  Arduino Uno R3
 * ============================================================
 *
 *  WIRING
 *  ------
 *  RC522 RFID  SDA=10  SCK=13  MOSI=11  MISO=12  RST=9  3.3V  GND
 *  Keypad      R1=A0  R2=A1  R3=A2  R4=A3  C1=5  C2=6  C3=8  C4=4
 *  LCD I2C     SDA=A4  SCL=A5   (address 0x27, change to 0x3F if blank)
 *  Switch      pin 3 to GND  (INPUT_PULLUP, LOW = admin mode)
 *  Relay       IN=7   (LOW = locked, HIGH = unlocked)
 *
 *  NORMAL MODE  (switch open / HIGH)
 *  0-9  type PIN digit        #  confirm PIN
 *  *    backspace / cancel    A  face recognition
 *  B    show door status      C  go to idle screen
 *  D    show last log entries
 *
 *  ADMIN MODE  (flip switch pin 3 to GND, then enter ADMIN_PIN + #)
 *  A  scroll up   B  scroll down   1-7  jump to item
 *  #  execute     *  back to menu
 *  Flip switch back to exit admin at any time.
 *
 *  SERIAL PROTOCOL  (9600 baud, newline terminated)
 *  Arduino->Pi : FACE | ENROLL:<n> | DELFACE:<n>
 *                LOG:<method>,<who>,<r> | GETLOG | PING
 *  Pi->Arduino : GRANT:<n> | DENY:<reason> | MSG:<l1>|<l2>
 *                CMD:UNLOCK | CMD:LOCK | CMD:STATUS | PONG
 * ============================================================
 */

#include <SPI.h>
#include <MFRC522.h>
#include <Keypad.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>

// ============================================================
//  CONFIGURE
// ============================================================
const char* ADMIN_PIN    = "1234";
const char* VALID_PINS[] = { "6767", "5678" };
const char* VALID_UIDS[] = { "F3-AO-8D-FC" };  // use Show UID menu to find real values

#define OPEN_MS      5000   // ms door stays unlocked after grant
#define MAX_FAILS    3      // failed attempts before lockout
#define LOCKOUT_MS   30000  // lockout duration ms
// ============================================================

// -- Pins
#define RFID_SS    10
#define RFID_RST    9
#define RELAY_PIN   7
#define SW_PIN      3    // admin switch (INPUT_PULLUP, LOW = admin)

// -- Keypad
const byte KR = 4, KC = 4;
char keys[KR][KC] = {
  {'1','2','3','A'},
  {'4','5','6','B'},
  {'7','8','9','C'},
  {'*','0','#','D'}
};
byte rowPins[KR] = {A0, A1, A2, A3};
byte colPins[KC] = {5, 6, 8, 4};    // C4 on pin 4

// -- Objects
MFRC522           rfid(RFID_SS, RFID_RST);
Keypad            kpad = Keypad(makeKeymap(keys), rowPins, colPins, KR, KC);
LiquidCrystal_I2C lcd(0x27, 16, 2);

// -- Door state
bool  doorLocked = true;
bool  lockedOut  = false;
unsigned long lockAt    = 0;
unsigned long lockoutAt = 0;
int   failCount  = 0;

// -- Mode state
// adminMode = true only when switch is LOW AND correct PIN has been entered.
// Flipping the switch HIGH always returns to normal mode immediately.
bool  adminMode   = false;
bool  pinVerified = false;
bool  swWasLow    = false;   // previous switch reading for edge detection
String adminBuf   = "";
int   menuSel     = 0;

// -- Serial buffer
String rxBuf = "";

// -- Admin menu
const char* MENU[] = {
  "Show Card UID",
  "Enroll Face",
  "Delete Face",
  "View Log",
  "Force Unlock",
  "Force Lock",
  "Exit Admin"
};
const int MENU_LEN = 7;

// ============================================================
//  HELPERS
// ============================================================
void lcdShow(const String& l1, const String& l2) {
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print(l1.substring(0, 16));
  lcd.setCursor(0, 1); lcd.print(l2.substring(0, 16));
}

String starMask(const String& s) {
  String m = "";
  for (unsigned i = 0; i < s.length(); i++) m += '*';
  return m;
}

String lockWait() {
  long rem = ((long)(lockoutAt + LOCKOUT_MS) - (long)millis()) / 1000;
  if (rem < 0) rem = 0;
  return "Wait " + String(rem) + "s";
}

void setLock(bool lock) {
  doorLocked = lock;
  digitalWrite(RELAY_PIN, lock ? LOW : HIGH);
  if (!lock) lockAt = millis();
}

String getUID() {
  String u = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    if (rfid.uid.uidByte[i] < 0x10) u += "0";
    u += String(rfid.uid.uidByte[i], HEX);
    if (i < rfid.uid.size - 1) u += "-";
  }
  u.toUpperCase();
  return u;
}

bool validPIN(const String& p) {
  int n = sizeof(VALID_PINS) / sizeof(VALID_PINS[0]);
  for (int i = 0; i < n; i++) if (p == String(VALID_PINS[i])) return true;
  return false;
}

bool validUID(const String& u) {
  int n = sizeof(VALID_UIDS) / sizeof(VALID_UIDS[0]);
  for (int i = 0; i < n; i++) if (u == String(VALID_UIDS[i])) return true;
  return false;
}

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(9600);
  Wire.begin();
  lcd.init();
  lcd.backlight();
  SPI.begin();
  rfid.PCD_Init();
  pinMode(RELAY_PIN, OUTPUT);
  pinMode(SW_PIN, INPUT_PULLUP);
  setLock(true);
  swWasLow = (digitalRead(SW_PIN) == LOW);
  lcdShow("Smart Lock", "Ready");
  delay(1500);
  showIdle();
}

// ============================================================
//  MAIN LOOP
// ============================================================
void loop() {
  readSerial();
  handleSwitch();
  checkTimers();

  if (adminMode) {
    doAdmin();
  } else {
    doNormal();
  }
}

// ============================================================
//  SWITCH  --  edge-detected, debounced
//
//  Key logic:
//    Falling edge (HIGH->LOW):  switch closed  -> ask for PIN
//    Rising edge  (LOW->HIGH):  switch opened  -> exit admin
//    adminMode is set TRUE only when PIN is verified, not on switch flip.
// ============================================================
void handleSwitch() {
  bool swLow = (digitalRead(SW_PIN) == LOW);

  if (swWasLow == swLow) return;   // no edge

  delay(30);   // debounce pause
  swLow = (digitalRead(SW_PIN) == LOW);
  if (swWasLow == swLow) return;   // was noise

  swWasLow = swLow;

  if (!swLow) {
    // Switch opened -> exit admin immediately, return to normal
    adminMode   = false;
    pinVerified = false;
    adminBuf    = "";
    showIdle();
  } else {
    // Switch closed -> prompt for admin PIN
    // adminMode stays false until PIN is verified
    adminMode   = false;
    pinVerified = false;
    adminBuf    = "";
    menuSel     = 0;
    lcdShow("Admin Mode", "PIN then #");
  }
}

// ============================================================
//  TIMERS
// ============================================================
void checkTimers() {
  if (!doorLocked && (millis() - lockAt >= OPEN_MS)) {
    setLock(true);
    if (!adminMode) showIdle();
  }
  if (lockedOut && (millis() - lockoutAt >= LOCKOUT_MS)) {
    lockedOut = false;
    failCount = 0;
    if (!adminMode) showIdle();
  }
}

// ============================================================
//  IDLE SCREEN
// ============================================================
void showIdle() {
  if (digitalRead(SW_PIN) == LOW && !adminMode) {
    lcdShow("Admin Mode", "PIN then #");
    return;
  }
  lcdShow("Scan/PIN/#=OK", "A=Face  D=Log");
}

// ============================================================
//  GRANT / DENY
// ============================================================
void grantAccess(const String& msg) {
  failCount = 0;
  setLock(false);
  lcd.clear();
  lcd.setCursor(0, 0); lcd.print("  ACCESS OK     ");
  lcd.setCursor(0, 1); lcd.print(msg.substring(0, 16));
  lcd.noBacklight(); delay(120); lcd.backlight();
  delay(120); lcd.noBacklight(); delay(120); lcd.backlight();
  delay(1500);
  if (!adminMode) showIdle();
}

void denyAccess(const String& msg) {
  failCount++;
  if (failCount >= MAX_FAILS) {
    lockedOut = true;
    lockoutAt = millis();
    failCount = 0;
    lcdShow("LOCKED OUT", "Wait 30s");
  } else {
    lcdShow("ACCESS DENIED", msg.substring(0, 16));
    lcd.noBacklight(); delay(150); lcd.backlight();
    delay(150); lcd.noBacklight(); delay(150); lcd.backlight();
  }
  delay(2000);
  if (!adminMode) showIdle();
}

// ============================================================
//  NORMAL MODE
// ============================================================
void doNormal() {
  // Check RFID
  if (rfid.PICC_IsNewCardPresent() && rfid.PICC_ReadCardSerial()) {
    String uid = getUID();
    rfid.PICC_HaltA();
    rfid.PCD_StopCrypto1();
    if (lockedOut) { lcdShow("LOCKED OUT", lockWait()); delay(2000); return; }
    lcdShow("Card:", uid.substring(0, 11));
    if (validUID(uid)) {
      grantAccess("Card accepted");
      Serial.println("LOG:RFID," + uid + ",OK");
    } else {
      denyAccess("Unknown card");
      Serial.println("LOG:RFID," + uid + ",DENY");
    }
    return;
  }

  char k = kpad.getKey();
  if (!k) return;

  static String pinBuf = "";

  if (k >= '0' && k <= '9') {
    if (pinBuf.length() < 8) {
      pinBuf += k;
      lcdShow("Enter PIN:", starMask(pinBuf));
    }
  } else if (k == '#') {
    if (pinBuf.length() == 0) return;
    if (lockedOut) { lcdShow("LOCKED OUT", lockWait()); pinBuf = ""; delay(2000); return; }
    if (validPIN(pinBuf)) {
      grantAccess("PIN accepted");
      Serial.println("LOG:PIN,user,OK");
    } else {
      denyAccess("Wrong PIN");
      Serial.println("LOG:PIN,user,DENY");
    }
    pinBuf = "";
  } else if (k == '*') {
    if (pinBuf.length() > 0) {
      pinBuf.remove(pinBuf.length() - 1);
      lcdShow("Enter PIN:", pinBuf.length() > 0 ? starMask(pinBuf) : "_");
    } else {
      showIdle();
    }
  } else if (k == 'A') {
    pinBuf = "";
    if (lockedOut) { lcdShow("LOCKED OUT", lockWait()); delay(2000); return; }
    lcdShow("Face Auth", "Look at camera");
    Serial.println("FACE");
    waitForPi(12000);
  } else if (k == 'B') {
    pinBuf = "";
    lcdShow(doorLocked ? "Door: LOCKED" : "Door: OPEN",
            lockedOut  ? "System LOCKED" : "System OK");
    delay(2500);
    showIdle();
  } else if (k == 'C') {
    pinBuf = "";
    showIdle();
  } else if (k == 'D') {
    pinBuf = "";
    Serial.println("GETLOG");
    waitForPi(14000);
    showIdle();
  }
}

// ============================================================
//  ADMIN MODE
// ============================================================
void doAdmin() {
  char k = kpad.getKey();
  if (!k) return;

  // PIN verification phase
  if (!pinVerified) {
    if (k >= '0' && k <= '9' && adminBuf.length() < 8) {
      adminBuf += k;
      lcdShow("Admin PIN:", starMask(adminBuf));
    } else if (k == '#') {
      if (adminBuf == String(ADMIN_PIN)) {
        pinVerified = true;
        menuSel = 0;
        showMenu();
      } else {
        lcdShow("Wrong PIN!", "Try again");
        adminBuf = "";
        delay(1500);
        lcdShow("Admin Mode", "PIN then #");
      }
    } else if (k == '*') {
      adminBuf = "";
      lcdShow("Admin Mode", "PIN then #");
    }
    return;
  }

  // Menu navigation phase
  if      (k == 'A')              { menuSel = (menuSel - 1 + MENU_LEN) % MENU_LEN; showMenu(); }
  else if (k == 'B')              { menuSel = (menuSel + 1) % MENU_LEN;             showMenu(); }
  else if (k >= '1' && k <= '7') { menuSel = k - '1';                               showMenu(); }
  else if (k == '#')              { runMenuItem(menuSel); }
  else if (k == '*')              { showMenu(); }
}

void showMenu() {
  String bar = "";
  for (int i = 0; i < MENU_LEN; i++) bar += (i == menuSel) ? '>' : '-';
  String top = bar + " " + String(menuSel + 1) + "/" + String(MENU_LEN);
  lcdShow(top.substring(0, 16), String(MENU[menuSel]).substring(0, 16));
}

// Collect digit string until # (confirm) or * (cancel), with timeout
String getDigits(unsigned long ms) {
  String s = "";
  unsigned long dl = millis() + ms;
  lcdShow("Enter:", "_");
  while (millis() < dl) {
    char k = kpad.getKey();
    if (!k) { delay(40); continue; }
    if (k >= '0' && k <= '9' && s.length() < 12) {
      s += k;
      lcdShow("Enter:", starMask(s));
    } else if (k == '#') {
      return s;
    } else if (k == '*') {
      if (s.length() > 0) {
        s.remove(s.length() - 1);
        lcdShow("Enter:", s.length() > 0 ? starMask(s) : "_");
      } else {
        return "";
      }
    }
    delay(40);
  }
  return "";
}

void runMenuItem(int idx) {
  switch (idx) {
    case 0: {  // Show card UID
      lcdShow("Scan a card", "10s timeout");
      unsigned long dl = millis() + 10000;
      bool found = false;
      while (millis() < dl && !found) {
        readSerial();
        if (rfid.PICC_IsNewCardPresent() && rfid.PICC_ReadCardSerial()) {
          String uid = getUID();
          rfid.PICC_HaltA(); rfid.PCD_StopCrypto1();
          lcdShow("UID:", uid.substring(0, 11));
          Serial.println("UID:" + uid);
          delay(5000);
          found = true;
        }
        delay(50);
      }
      if (!found) { lcdShow("No card", "timed out"); delay(1500); }
      showMenu();
      break;
    }
    case 1: {  // Enroll face
      lcdShow("Enter name:", "digits + #");
      String name = getDigits(15000);
      if (name.length() > 0) {
        lcdShow("Starting...", "Look at camera");
        Serial.println("ENROLL:" + name);
        waitForPi(40000);
      } else {
        lcdShow("Cancelled", ""); delay(1000);
      }
      showMenu();
      break;
    }
    case 2: {  // Delete face
      lcdShow("Enter name:", "digits + #");
      String name = getDigits(15000);
      if (name.length() > 0) {
        Serial.println("DELFACE:" + name);
        waitForPi(5000);
      } else {
        lcdShow("Cancelled", ""); delay(1000);
      }
      showMenu();
      break;
    }
    case 3: {  // View log
      Serial.println("GETLOG");
      waitForPi(15000);
      showMenu();
      break;
    }
    case 4: {  // Force unlock
      setLock(false);
      lcdShow("Unlocked!", "Admin override");
      Serial.println("LOG:ADMIN,admin,UNLOCK");
      delay(2000);
      showMenu();
      break;
    }
    case 5: {  // Force lock
      setLock(true);
      lcdShow("Locked!", "");
      delay(1500);
      showMenu();
      break;
    }
    case 6: {  // Exit
      lcdShow("Flip switch", "to exit admin");
      break;
    }
  }
}

// ============================================================
//  PI SERIAL
// ============================================================
void readSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      rxBuf.trim();
      if (rxBuf.length() > 0) handlePiMsg(rxBuf);
      rxBuf = "";
    } else if (c != '\r') {
      rxBuf += c;
    }
  }
}

void handlePiMsg(const String& msg) {
  if (msg.startsWith("GRANT:")) {
    String name = msg.substring(6); name.trim();
    failCount = 0;
    setLock(false);
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("  ACCESS OK     ");
    lcd.setCursor(0, 1); lcd.print(("Hi " + name).substring(0, 16));
    lcd.noBacklight(); delay(120); lcd.backlight();
    delay(120); lcd.noBacklight(); delay(120); lcd.backlight();
    delay(1500);
    if (!adminMode) showIdle();
  } else if (msg.startsWith("DENY:")) {
    String reason = msg.substring(5); reason.trim();
    failCount++;
    if (failCount >= MAX_FAILS) {
      lockedOut = true; lockoutAt = millis(); failCount = 0;
      lcdShow("LOCKED OUT", "Wait 30s");
    } else {
      lcdShow("ACCESS DENIED", reason.substring(0, 16));
      lcd.noBacklight(); delay(150); lcd.backlight();
      delay(150); lcd.noBacklight(); delay(150); lcd.backlight();
    }
    delay(2000);
    if (!adminMode) showIdle();
  } else if (msg.startsWith("MSG:")) {
    int sep = msg.indexOf('|');
    String l1 = (sep > 0) ? msg.substring(4, sep) : msg.substring(4);
    String l2 = (sep > 0) ? msg.substring(sep + 1) : "";
    lcdShow(l1.substring(0, 16), l2.substring(0, 16));
    delay(2500);
    if (adminMode && pinVerified) showMenu();
    else if (!adminMode) showIdle();
  } else if (msg.startsWith("CMD:")) {
    String cmd = msg.substring(4); cmd.trim();
    if      (cmd == "UNLOCK") { setLock(false); lcdShow("Remote Unlock", "via web"); }
    else if (cmd == "LOCK")   { setLock(true);  lcdShow("Remote Lock",   "via web"); }
    else if (cmd == "STATUS") { Serial.println(doorLocked ? "STATUS:LOCKED" : "STATUS:UNLOCKED"); }
  } else if (msg == "PONG") { /* heartbeat */ }
}

// Block until Pi sends one reply or timeout expires
void waitForPi(unsigned long ms) {
  unsigned long dl = millis() + ms;
  String buf = "";
  while (millis() < dl) {
    while (Serial.available()) {
      char c = Serial.read();
      if (c == '\n') {
        buf.trim();
        if (buf.length() > 0) handlePiMsg(buf);
        return;
      } else if (c != '\r') {
        buf += c;
      }
    }
    delay(10);
  }
}
