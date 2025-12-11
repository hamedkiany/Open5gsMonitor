#include <ESP8266WiFi.h>
#include <WiFiClientSecure.h>
#include <UniversalTelegramBot.h>
#include <WiFiClient.h>
#include <ESP8266HTTPClient.h>
#include "login.h"  // WIFI_SSID, WIFI_PASSWORD, BOT_TOKEN, TELEGRAM_CERTIFICATE_ROOT

// ================== TELEGRAM ==================
X509List cert(TELEGRAM_CERTIFICATE_ROOT);
WiFiClientSecure secured_client;
UniversalTelegramBot bot(BOT_TOKEN, secured_client);

// Chat ID al que quieres enviar las notificaciones
const char* SERVICE_CHAT_ID = "308137514";

// ================== SERVICIOS OPEN5GS ==================
struct ServiceStatus {
  const char* name;
  bool up;     // true = UP, false = DOWN
  bool valid;  // false = aún no hemos visto este servicio
};

ServiceStatus services[] = {
  { "NRF",  true, false },
  { "SCP",  true, false },
  { "UPF",  true, false },
  { "SMF",  true, false },
  { "AMF",  true, false },
  { "AUSF", true, false },
  { "UDM",  true, false },
  { "PCF",  true, false },
  { "NSSF", true, false },
  { "BSF",  true, false },
  { "UDR",  true, false },
};

const size_t NUM_SERVICES = sizeof(services) / sizeof(services[0]);

// API donde lees el estado
const char* API_URL = "http://1.ibuild.es:8000/events?limit=50";

// Poll cada 2 segundos
const unsigned long SERVICE_POLL_INTERVAL_MS = 2000;
unsigned long lastServicePoll = 0;

// ================== WIFI ==================
void connectWiFi() {
  Serial.print("Connecting to Wifi ");
  Serial.println(WIFI_SSID);

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  while (WiFi.status() != WL_CONNECTED) {
    Serial.print(".");
    delay(500);
  }

  Serial.print("\nConnected. IP: ");
  Serial.println(WiFi.localIP());

  // Necesario para validar el certificado de Telegram
  configTime(0, 0, "pool.ntp.org");
  secured_client.setTrustAnchors(&cert);
}

// ================== LÓGICA DE ESTADOS ==================
void updateServiceStatus(const String& name, const String& stateRaw) {
  String state = stateRaw;
  String chat_id2 = "308137514";
  state.trim();
  state.toUpperCase();

  bool isUp;
  if (state == "UP")       isUp = true;
  else if (state == "DOWN") isUp = false;
  else {
    //Serial.print("Estado desconocido para ");
    //Serial.print(name);
    //Serial.print(": ");
    //Serial.println(state);
    delay(20);
    return;
  }

  for (size_t i = 0; i < NUM_SERVICES; i++) {
    if (name == services[i].name) {
      // Primera vez que vemos este servicio → solo guardamos estado
      if (!services[i].valid) {
        services[i].up = isUp;
        services[i].valid = true;
        //Serial.print("Inicial ");
        //Serial.print(name);
        //Serial.print(" = ");
        Serial.println(isUp ? "UP" : "DOWN");
      } else if (services[i].up != isUp) {
        // Cambio real
        services[i].up = isUp;

        String msg = "[SERVICE_CHANGE] ";
        msg += services[i].name;
        msg += "=";
        msg += (services[i].up ? "UP" : "DOWN");

        Serial.print("CAMBIO -> ");
        Serial.println(msg);
        bot.sendMessage(chat_id2, msg, "");
      } else {
        // Sin cambio (opcional, solo debug)
        //Serial.print("Sin cambio ");
        //Serial.print(name);
        //Serial.print(" sigue ");
        //Serial.println(isUp ? "UP" : "DOWN");
      }
      break;
    }
  }
}

void parseServiceLine(const String& line) {
  int bracketPos = line.indexOf(']');
  String rest = (bracketPos >= 0) ? line.substring(bracketPos + 1) : line;
  rest.trim();

  int start = 0;
  while (start < rest.length()) {
    int spacePos = rest.indexOf(' ', start);
    String token = (spacePos == -1) ? rest.substring(start)
                                    : rest.substring(start, spacePos);

    token.trim();
    if (token.length() > 0) {
      int eqPos = token.indexOf('=');
      if (eqPos > 0) {
        String name  = token.substring(0, eqPos);
        String state = token.substring(eqPos + 1);

        name.trim();
        state.trim();

       // Serial.print("Token -> '");
        //Serial.print(name);
        //Serial.print("' = '");
        //Serial.print(state);
        //Serial.println("'");
        delay(20);
        updateServiceStatus(name, state);
      }
    }

    if (spacePos == -1) break;
    start = spacePos + 1;
  }
}

void pollApiForServiceStatus() {
//  if (WiFi.status() != WL_CONNECTED) {
//    Serial.println("WiFi NOT connected, skipping API poll");
//    return;
//  }

  WiFiClient client;
  HTTPClient http;

  Serial.println("HTTP GET eventos...");
  if (!http.begin(client, API_URL)) {
    Serial.println("http.begin() failed");
    return;
  }

  int httpCode = http.GET();

  if (httpCode == HTTP_CODE_OK) {
    String payload = http.getString();

    int idx = payload.lastIndexOf("[SERVICE]");
    if (idx >= 0) {
      int end = payload.indexOf('\n', idx);
      String serviceLine = (end > idx) ? payload.substring(idx, end)
                                       : payload.substring(idx);
      serviceLine.trim();

      //Serial.print("SERVICE LINE = ");
      //Serial.println(serviceLine);
      delay(20);
      parseServiceLine(serviceLine);
    } else {
      delay(20);
      //Serial.println("No [SERVICE] line found in payload");
    }
  } else {
    delay(20);
    //Serial.printf("HTTP error: %d\n", httpCode);
  }

  http.end();
}

// ================== SETUP / LOOP ==================
void setup() {
  Serial.begin(115200);
  Serial.println();
  connectWiFi();
  Serial.println("Arrancado, esperando cambios de servicios...");
}

void loop() {
  unsigned long now = millis();

  // cada 2 segundos leemos el estado de servicio
  if (now - lastServicePoll > SERVICE_POLL_INTERVAL_MS) {
    lastServicePoll = now;
    pollApiForServiceStatus();
  }
}
