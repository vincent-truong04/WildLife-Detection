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
const size_t   CHUNK_SIZE            = 1024;
const uint32_t WRITE_RETRY_DELAY_MS  = 100;
const int      WRITE_RETRY_MAX       = 3;
const uint32_t CONNECT_TIMEOUT_MS    = 20000;
const uint32_t ACK_TIMEOUT_MS        = 25000;
const uint32_t RECONNECT_BASE_DELAY  = 1000;
const uint32_t RECONNECT_MAX_DELAY   = 30000;
const int      RECONNECT_MAX         = 8;
const uint32_t SOCKET_TEARDOWN_MS    = 2000;
const uint32_t HALOW_LINK_TIMEOUT    = 60000;
const uint32_t COOLDOWN_MS        = 65000;
const int      SEND_RETRIES       = 5;
const uint32_t PRE_CAPTURE_DELAY_MS = 400;   // let animal move into frame
const int      BURST_COUNT           = 3;    // frames per motion event
const uint32_t BURST_INTERVAL_MS     = 1500;  // gap between burst frames


const uint32_t HEARTBEAT_INTERVAL_MS = 20 * 1000;

// Ignore PIR for this long after any TX, to reject RF-induced false edges
const uint32_t PIR_BLANK_AFTER_TX_MS = 3000;
unsigned long g_last_tx_ms = 0;

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
    yield();
    delay(2);   // let radio drain between chunks
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

// Read up to N bytes within timeout. Returns how many were read.
int read_bytes(uint8_t* buf, int n, uint32_t timeout_ms) {
  unsigned long deadline = millis() + timeout_ms;
  int got = 0;
  while (got < n && millis() < deadline) {
    if (g_client.available()) {
      buf[got++] = g_client.read();
    } else if (!g_client.connected()) {
      Serial.println("[TCP] Connection dropped while reading");
      return got;
    } else {
      delay(5);
    }
  }
  return got;
}

bool do_handshake() {
  Serial.printf("[Handshake] Identifying as camera '%s'\n", CAM_ID);

  // Drain anything sitting in RX before we start
  delay(50);
  int drained = 0;
  while (g_client.available()) {
    uint8_t b = g_client.read();
    Serial.printf("[Handshake] Drained pre-byte #%d: 0x%02X\n", ++drained, b);
    if (drained > 64) break;  // safety
  }

  uint8_t id_len = (uint8_t)strlen(CAM_ID);
  Serial.printf("[Handshake] Sending id_len=0x%02X then '%s'\n", id_len, CAM_ID);
  if (!write_all(&id_len, 1))                     return false;
  if (!write_all((const uint8_t*)CAM_ID, id_len)) return false;
  g_client.flush();

  // Read up to 8 bytes — if the Pi sends just ACK, we'll get one byte and
  // the rest of the buffer stays empty. If something weird is happening,
  // we'll see it.
  uint8_t resp[8] = {0};
  int n = read_bytes(resp, 8, CONNECT_TIMEOUT_MS);

  Serial.printf("[Handshake] Got %d byte(s):", n);
  for (int i = 0; i < n; i++) Serial.printf(" 0x%02X", resp[i]);
  Serial.println();

  if (n >= 1 && resp[0] == ACK_BYTE) {
    Serial.println("[Handshake] Pi accepted ✓");
    // If extra bytes came in, that's a bug we want to know about
    if (n > 1) {
      Serial.printf("[Handshake] WARNING: %d extra byte(s) after ACK\n", n - 1);
    }
    return true;
  }

  // Detect TLS Alert record — something in the HaLow stack is injecting
  // synthetic responses when the real connection isn't established yet.
  // Signature: first byte 0x15, followed by 0x03 0x03 (TLS 1.2 version).
  if (n >= 3 && resp[0] == 0x15 && resp[1] == 0x03 && resp[2] == 0x03) {
    Serial.println("[Handshake] Phantom TLS-alert response — radio stack "
                   "not ready, waiting before retry");
    g_tcp_live = false;
    delay(3000);
    return false;
  }

  Serial.printf("[Handshake] Pi rejected (first byte 0x%02X): will retry\n",
                n >= 1 ? resp[0] : 0xFF);
  g_tcp_live = false;
  return false;
}

bool ensure_tcp_connected() {
  if (g_tcp_live && g_client.connected()) return true;

  // Always tear down, regardless of what connected() says
  Serial.println("[TCP] Forcing socket teardown before reconnect");
  g_client.stop();
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

    Serial.println("[TCP] connect() succeeded — starting handshake");
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
  Serial.printf("\n[Cam %s] Motion detected: burst of %d\n", CAM_ID, BURST_COUNT);

  // Wait for the animal to move into the centre of the frame
  delay(PRE_CAPTURE_DELAY_MS);

  for (int shot = 1; shot <= BURST_COUNT; shot++) {
    Serial.printf("[Cam %s] Burst frame %d/%d\n", CAM_ID, shot, BURST_COUNT);

    camera_fb_t* fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("[Cam] Capture failed: no frame");
      if (shot < BURST_COUNT) delay(BURST_INTERVAL_MS);
      continue;
    }

    bool     ok      = false;
    uint32_t backoff = 1000;

    for (int attempt = 1; attempt <= SEND_RETRIES && !ok; attempt++) {
      if (!ensure_tcp_connected()) break;
      ok = send_image(fb->buf, fb->len);
      if (!ok && attempt < SEND_RETRIES) {
        delay(backoff);
        backoff = min(backoff * 2, (uint32_t)30000);
      }
    }

    Serial.printf("[Cam %s] Frame %d: %s\n", CAM_ID, shot,
                  ok ? "OK" : "FAILED");
    esp_camera_fb_return(fb);

    if (shot < BURST_COUNT) delay(BURST_INTERVAL_MS);
  }
}


void setup() {
  Serial.begin(115200);
  
  // PIR needs ~60 s to stabilise after power-on
  pinMode(PIR_PIN, INPUT);


  Serial.println("[PIR] Waiting for sensor to stabilise...");
  for (int i = 5; i > 0; i--) {
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
  config.jpeg_quality = 14;
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
  s->set_sharpness(s, 2);          // crisper edges on fur/feathers
  s->set_denoise(s, 1);            // reduces grain in low-light shots
  s->set_exposure_ctrl(s, 1);      // enable auto-exposure
  s->set_aec2(s, 1);               // AEC DSP — better exposure in high-contrast scenes
  s->set_gain_ctrl(s, 1);          // enable auto-gain
  s->set_awb_gain(s, 1);           // auto white-balance gain
  s->set_lenc(s, 1);               // lens correction for even illumination

  camera_ready = true;
  Serial.printf("[Cam %s] Camera ready\n", CAM_ID);

  HaLow.init("US");
  HaLow.config(LOCAL_IP, GATEWAY, SUBNET, DNS);
  halow_connect();

  ensure_tcp_connected();
}


// Returns true if PIR stays HIGH for `required_ms` of consecutive sampling
bool pir_confirmed(uint32_t required_ms) {
  unsigned long start = millis();
  while (millis() - start < required_ms) {
    if (digitalRead(PIR_PIN) != HIGH) return false;
    delay(5);
  }
  return true;
}

void loop() {
  static bool pir_prev = false;

  bool pir_now      = digitalRead(PIR_PIN) == HIGH;
  bool rising_edge  = pir_now && !pir_prev;
  bool cooled_down  = (millis() - g_last_trigger_ms) >= COOLDOWN_MS;
  bool tx_quiet     = (millis() - g_last_tx_ms) >= PIR_BLANK_AFTER_TX_MS;

  if (rising_edge && cooled_down && tx_quiet && !upload_active) {
    if (pir_confirmed(300)) {
      Serial.printf("[PIR] Motion CONFIRMED @ t=%lu (since_last=%lu ms)\n",
                    millis(), millis() - g_last_trigger_ms);
      g_last_trigger_ms = millis();
      upload_active     = true;
      capture_and_send();
      upload_active     = false;
      g_last_tx_ms      = millis();
    } else {
      Serial.println("[PIR] Glitch rejected (failed 300ms hold)");
    }
  }

  if (tx_quiet) pir_prev = pir_now;

  // HEARTBEAT DISABLED FOR DEBUGGING — re-enable after testing
  // if ((millis() - g_last_sent_ms) >= HEARTBEAT_INTERVAL_MS && !upload_active) {
  //   Serial.println("[HB] === heartbeat firing ===");
  //   ensure_tcp_connected();
  //   send_heartbeat();
  //   g_last_tx_ms = millis();
  // }
}
