#include "esp_camera.h"
#include <HaLow.h>



//  PER-UNIT CONFIGURATION
const char*  CAM_ID   = "A";
const int    PIR_PIN  = 1;
IPAddress    LOCAL_IP(192, 168, 100, 116);   // unique per unit


// Network
const char*  SSID     = "Wildlife_Halow";
const char*  PASSWORD = "heltec.org";
const char*  HOST     = "192.168.100.1";   // Raspberry Pi HaLow IP
const int    PORT     = 8080;
IPAddress    GATEWAY(192, 168, 100, 1);
IPAddress    SUBNET(255, 255, 255, 0);
IPAddress    DNS(192, 168, 100, 1);


// Timing & flow-control 
const size_t   CHUNK_SIZE            = 2048;
const uint32_t WRITE_RETRY_DELAY_MS  = 150;
const int      WRITE_RETRY_MAX       = 10;
const uint32_t CONNECT_TIMEOUT_MS    = 20000;
const uint32_t ACK_TIMEOUT_MS        = 25000;
const uint32_t RECONNECT_BASE_DELAY  = 1000;
const uint32_t RECONNECT_MAX_DELAY   = 30000;
const int      RECONNECT_MAX         = 8;
const uint32_t SOCKET_TEARDOWN_MS    = 2000;
const uint32_t HALOW_LINK_TIMEOUT    = 60000;
const uint32_t COOLDOWN_MS           = 12000;
const int      SEND_RETRIES          = 5;
const uint32_t HEARTBEAT_INTERVAL_MS = 20 * 1000;


// Protocol Bytes
const uint8_t ACK_BYTE = 0xAC;
const uint8_t NAK_BYTE = 0x00;


// Runtime state
bool          camera_ready      = false;
bool          upload_active     = false;
unsigned long g_last_trigger_ms = 0;

HalowClient   g_client;
bool          g_tcp_live        = false;
unsigned long g_last_sent_ms    = 0;

// Block until HaLow associates or timeout
void halow_connect() {
  if (HaLow.status() == WL_CONNECTED) return;

  Serial.printf("\n[HaLow] Connecting to '%s'…\n", SSID);
  HaLow.begin(SSID, PASSWORD);

  unsigned long start = millis();
  while (HaLow.status() != WL_CONNECTED) {
    if (millis() - start > HALOW_LINK_TIMEOUT) {
      Serial.println("[HaLow] Timed out — rebooting");
      ESP.restart();
    }
    delay(500);
    Serial.print(".");
  }

  Serial.printf("\n[HaLow] Associated  IP:%s  GW:%s\n",
                HaLow.localIP().toString().c_str(),
                HaLow.gatewayIP().toString().c_str());
}

// Send exactly len bytes, retrying stalled chunks up to WRITE_RETRY_MAX times.
bool write_all(const uint8_t* buf, size_t len) {
  size_t sent = 0;

  while (sent < len) {
    if (!g_client.connected()) {
      Serial.printf("[TCP] Connection dropped after %d/%d bytes\n", sent, len);
      g_tcp_live = false;
      return false;
    }

    size_t to_send = min(CHUNK_SIZE, len - sent);
    size_t written = 0;

    for (int stall = 0; stall <= WRITE_RETRY_MAX; stall++) {
      written = g_client.write(buf + sent, to_send);
      if (written > 0) break;
      Serial.printf("[TCP] TX stall at byte %d/%d (retry %d/%d)\n",
                    sent, len, stall + 1, WRITE_RETRY_MAX);
      delay(WRITE_RETRY_DELAY_MS);
    }

    if (written == 0) {
      Serial.printf("[TCP] Write permanently stalled at byte %d/%d\n", sent, len);
      g_tcp_live = false;
      return false;
    }

    sent += written;
  }
  return true;
}

// Wait for one byte from the Pi. Returns the byte or -1
int wait_for_byte(uint32_t timeout_ms) {
  unsigned long deadline = millis() + timeout_ms;

  while (!g_client.available()) {
    if (!g_client.connected()) {
      Serial.println("[TCP] Connection closed while waiting for byte");
      g_tcp_live = false;
      return -1;
    }
    if (millis() > deadline) {
      Serial.printf("[TCP] Timed out waiting for byte (%u ms)\n", timeout_ms);
      return -1;
    }
    delay(5);
  }
  return g_client.read();
}

// Send camera ID and wait for Pi acceptance
bool do_handshake() {
  Serial.printf("[Handshake] Identifying as camera '%s'\n", CAM_ID);

  uint8_t id_len = (uint8_t)strlen(CAM_ID);
  if (!write_all(&id_len, 1))                      return false;
  if (!write_all((const uint8_t*)CAM_ID, id_len))  return false;

  int response = wait_for_byte(CONNECT_TIMEOUT_MS);
  if (response == ACK_BYTE) {
    Serial.println("[Handshake] Pi accepted ✓");
    return true;
  }

  Serial.printf("[Handshake] Pi rejected (0x%02X): will retry\n",
                (uint8_t)response);
  g_tcp_live = false;
  return false;
}

// Ensure the TCP session is live. Reboots after RECONNECT_MAX consecutive failures.
bool ensure_tcp_connected() {
  if (g_tcp_live && g_client.connected()) return true;

  if (g_client.connected()) {
    Serial.println("[TCP] Closing stale socket before reconnect");
    g_client.stop();
  }
  delay(SOCKET_TEARDOWN_MS);
  g_tcp_live = false;

  uint32_t backoff = RECONNECT_BASE_DELAY;

  for (int attempt = 1; attempt <= RECONNECT_MAX; attempt++) {
    Serial.printf("[TCP] Reconnect attempt %d/%d → %s:%d\n",
                  attempt, RECONNECT_MAX, HOST, PORT);

    halow_connect();

    g_client.setTimeout(CONNECT_TIMEOUT_MS);
    if (!g_client.connect(HOST, PORT)) {
      Serial.printf("[TCP] connect() failed: backing off %u ms\n", backoff);
      g_client.stop();
      delay(SOCKET_TEARDOWN_MS);
      delay(backoff);
      backoff = min(backoff * 2, RECONNECT_MAX_DELAY);
      continue;
    }

    if (!do_handshake()) {
      g_client.stop();
      delay(SOCKET_TEARDOWN_MS);
      delay(backoff);
      backoff = min(backoff * 2, RECONNECT_MAX_DELAY);
      continue;
    }

    g_tcp_live     = true;
    g_last_sent_ms = millis();
    Serial.println("[TCP] Persistent session established");
    return true;
  }

  Serial.println("[TCP] All reconnect attempts failed: rebooting");
  delay(1000);
  ESP.restart();
  return false;
}

// Send [4B length header] + [JPEG payload], then wait for ACK.
bool send_image(const uint8_t* buf, uint32_t len) {
  uint8_t len_buf[4] = {
    (uint8_t)( len        & 0xFF),
    (uint8_t)((len >>  8) & 0xFF),
    (uint8_t)((len >> 16) & 0xFF),
    (uint8_t)((len >> 24) & 0xFF),
  };

  if (!write_all(len_buf, 4)) return false;

  Serial.printf("[Cam %s] Streaming %u bytes…\n", CAM_ID, len);
  if (!write_all(buf, len)) return false;

  g_client.flush();
  Serial.println("[TCP] Transfer complete: awaiting ACK");

  int response = wait_for_byte(ACK_TIMEOUT_MS);
  if (response == ACK_BYTE) {
    Serial.println("[ACK] ✓");
    g_last_sent_ms = millis();
    return true;
  }

  if (response == NAK_BYTE) {
    Serial.println("[NAK] Pi reported receive error: will retry image");
    return false;
  }

  Serial.printf("[ACK] Unexpected response 0x%02X: marking connection dead\n",
                (uint8_t)response);
  g_tcp_live = false;
  return false;
}

// Send a zero-length frame to keep the session alive.
void send_heartbeat() {
  if (!g_tcp_live) return;

  Serial.println("[HB] Sending heartbeat");

  uint8_t zero[4] = {0, 0, 0, 0};
  if (!write_all(zero, 4)) {
    Serial.println("[HB] Heartbeat failed: connection dead");
    return;
  }

  int response = wait_for_byte(ACK_TIMEOUT_MS);
  if (response == ACK_BYTE) {
    Serial.println("[HB] Ping");
    g_last_sent_ms = millis();
  } else {
    Serial.println("[HB] No ping: marking connection dead");
    g_tcp_live = false;
  }
}

// Capture a frame and upload it with retry.
void capture_and_send() {
  Serial.printf("\n[Cam %s] Motion detected: capturing\n", CAM_ID);

  camera_fb_t* fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("[Cam] Capture failed: no frame returned");
    return;
  }

  bool     ok      = false;
  uint32_t backoff = 1000;

  for (int attempt = 1; attempt <= SEND_RETRIES && !ok; attempt++) {
    Serial.printf("[Cam %s] Send attempt %d/%d  (%u bytes)\n",
                  CAM_ID, attempt, SEND_RETRIES, fb->len);

    if (!ensure_tcp_connected()) break;
    ok = send_image(fb->buf, fb->len);

    if (!ok && attempt < SEND_RETRIES) {
      Serial.printf("[Cam %s] Backing off %u ms before retry\n", CAM_ID, backoff);
      delay(backoff);
      backoff = min(backoff * 2, (uint32_t)30000);
    }
  }

  Serial.printf("[Cam %s] %s\n", CAM_ID,
                ok ? "Upload OK" : "Upload FAILED after all retries");

  esp_camera_fb_return(fb);
}


void setup() {
  Serial.begin(115200);
  
  // PIR needs ~60 s to stabilise after power-on
  pinMode(PIR_PIN, INPUT_PULLDOWN);
  gpio_pulldown_en((gpio_num_t)PIR_PIN);
  gpio_pullup_dis((gpio_num_t)PIR_PIN);


  Serial.println("[PIR] Waiting for sensor to stabilise...");
  for (int i = 60; i > 0; i--) {
    Serial.printf("[PIR] %d seconds remaining...\n", i);
    delay(1000);
  }

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_XGA;
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 10;
  config.grab_mode    = CAMERA_GRAB_LATEST;
  config.fb_count     = 2;

  if (esp_camera_init(&config) != ESP_OK) {
    Serial.println("[Cam] Init failed: halting");
    while (true) delay(1000);
  }

  sensor_t* s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_brightness(s, 1);
  s->set_saturation(s, 0);

  camera_ready = true;
  Serial.printf("[Cam %s] Camera ready\n", CAM_ID);

  HaLow.init("US");
  HaLow.config(LOCAL_IP, GATEWAY, SUBNET, DNS);
  halow_connect();

  ensure_tcp_connected();
}


void loop() {
  bool pir_high    = digitalRead(PIR_PIN) == HIGH;
  bool cooled_down = (millis() - g_last_trigger_ms) >= COOLDOWN_MS;

  // Capture and upload on motion, subject to PIR cooldown
  if (pir_high && cooled_down && !upload_active) {
    g_last_trigger_ms = millis();
    upload_active     = true;
    capture_and_send();
    upload_active     = false;
  }
  
  // Send a heartbeat when idle to prevent the session timing out
  if ((millis() - g_last_sent_ms) >= HEARTBEAT_INTERVAL_MS && !upload_active) {
    ensure_tcp_connected();
    send_heartbeat();
  }
}
