#include <WiFi.h>

const char* ssid = "jingjun_ubuntu";
const char* password = "12345678";

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Connecting to WiFi...");
  WiFi.mode(WIFI_STA);
  WiFi.disconnect(true);   // clear any stale config
  delay(100);
  WiFi.begin(ssid, password);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal strength (RSSI): ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println("\nFailed to connect to WiFi.");
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi still connected.");
  } else {
    Serial.println("WiFi disconnected. Reconnecting...");
    Serial.print("status=");
    Serial.println(WiFi.status());
    WiFi.reconnect();
  }
  delay(5000);
}
