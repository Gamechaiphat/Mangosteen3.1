#include <Arduino.h>
#include <WiFi.h>
#include "esp_camera.h"
#include "esp_http_server.h"
#include <DNSServer.h>
#include "esp_system.h"
#include "config.h"

// ==============================================================================
// 1. TFLITE MICRO SETUP
// ==============================================================================
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"

constexpr int kTensorArenaSize = 4*1024*1024;
uint8_t *tensor_arena = nullptr;
tflite::MicroInterpreter *interpreter = nullptr;
TfLiteTensor *input = nullptr;
TfLiteTensor *output = nullptr;

const char *class_names[] = {"overripe", "ripe", "unripe"};

struct InferenceResult
{
  char best_class[16];
  float probabilities[3]; // [0] = overripe, [1] = ripe, [2] = unripe
};
InferenceResult latest_result = {"Waiting...", {0.0f, 0.0f, 0.0f}};

// ==============================================================================
// 2. WEB SERVER & HTML
// ==============================================================================
static const char INDEX_HTML[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BALL-CAM Mangosteen AI</title>
  <style>
    body { font-family: sans-serif; background: #121212; color: #fff; text-align: center; margin: 0; padding: 20px; }
    .card { background: #1e1e1e; max-width: 400px; margin: auto; border-radius: 12px; padding: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.5); }
    img { width: 100%; height: auto; border-radius: 8px; border: 2px solid #333; background: #000; }
    .pred { font-size: 1.5rem; font-weight: bold; color: #4CAF50; margin: 15px 0 5px 0; text-transform: uppercase; }
    .bar-wrap { background: #333; border-radius: 6px; overflow: hidden; height: 16px; margin: 4px 0 10px 0; }
    .bar { height: 100%; width: 0%; transition: width 0.2s; }
    .label { display: flex; justify-content: space-between; font-size: 0.85rem; color: #ccc; }
    .stats { font-size: 0.75rem; color: #777; margin-top: 15px; border-top: 1px solid #333; padding-top: 10px; }
  </style>
</head>
<body>
  <div class="card">
    <h3>Mangosteen Group15</h3>
    <img id="stream" src="http://192.168.4.1:81/stream" alt="Live Feed">
    <div id="pred" class="pred">Waiting...</div>

    <!-- ลำดับที่ 1: Unripe (สีเหลือง) -->
    <div class="label"><span>Unripe</span><span id="p-unripe">0%</span></div>
    <div class="bar-wrap"><div id="b-unripe" class="bar" style="background: #FFC107;"></div></div>

    <!-- ลำดับที่ 2: Ripe (สีเขียว) -->
    <div class="label"><span>Ripe</span><span id="p-ripe">0%</span></div>
    <div class="bar-wrap"><div id="b-ripe" class="bar" style="background: #4CAF50;"></div></div>

    <!-- ลำดับที่ 3: Overripe (สีแดง) -->
    <div class="label"><span>Overripe</span><span id="p-overripe">0%</span></div>
    <div class="bar-wrap"><div id="b-overripe" class="bar" style="background: #F44336;"></div></div>

    <div class="stats" id="stats">AI Status: Active</div>
  </div>

  <script>
    setInterval(() => {
      fetch('/result?_=' + Date.now(), { cache: 'no-store' })
        .then(async r => {
          if (!r.ok) throw new Error('HTTP ' + r.status);
          return r.json();
        })
        .then(d => {
          document.getElementById('pred').innerText = d.class;
          
          // อัปเดตแถบ Unripe
          document.getElementById('p-unripe').innerText = d.unripe.toFixed(1) + '%';
          document.getElementById('b-unripe').style.width = d.unripe + '%';
          
          // อัปเดตแถบ Ripe
          document.getElementById('p-ripe').innerText = d.ripe.toFixed(1) + '%';
          document.getElementById('b-ripe').style.width = d.ripe + '%';
          
          // อัปเดตแถบ Overripe
          document.getElementById('p-overripe').innerText = d.overripe.toFixed(1) + '%';
          document.getElementById('b-overripe').style.width = d.overripe + '%';
        })
        .catch(e => {
          console.log('Fetch /result failed:', e);
        });
    }, 1000);
  </script>
</body>
</html>
)rawliteral";

static esp_err_t index_handler(httpd_req_t *req)
{
  httpd_resp_set_type(req, "text/html");
  return httpd_resp_send(req, INDEX_HTML, strlen(INDEX_HTML));
}

esp_err_t captive_portal_handler(httpd_req_t *req, httpd_err_code_t err)
{
  httpd_resp_set_status(req, "302 Found");
  httpd_resp_set_hdr(req, "Location", "http://192.168.4.1/");
  httpd_resp_send(req, NULL, 0);
  return ESP_OK;
}

static esp_err_t result_handler(httpd_req_t *req)
{
  char json[256];
  snprintf(json, sizeof(json),
           "{\"class\":\"%s\",\"overripe\":%.1f,\"ripe\":%.1f,\"unripe\":%.1f}",
           latest_result.best_class,
           latest_result.probabilities[0] * 100.0f,
           latest_result.probabilities[1] * 100.0f,
           latest_result.probabilities[2] * 100.0f);

  httpd_resp_set_type(req, "application/json");
  httpd_resp_set_hdr(req, "Access-Control-Allow-Origin", "*");
  httpd_resp_set_hdr(req, "Cache-Control", "no-store");
  return httpd_resp_send(req, json, strlen(json));
}

static esp_err_t stream_handler(httpd_req_t *req)
{
  camera_fb_t *fb = NULL;
  esp_err_t res = ESP_OK;
  size_t _jpg_buf_len = 0;
  uint8_t *_jpg_buf = NULL;
  char part_buf[128];

  res = httpd_resp_set_type(req, "multipart/x-mixed-replace;boundary=123456789000000000000987654321");
  if (res != ESP_OK)
    return res;

  while (true)
  {
    fb = esp_camera_fb_get();
    if (!fb)
    {
      res = ESP_FAIL;
    }
    else
    {
      _jpg_buf_len = fb->len;
      _jpg_buf = fb->buf;
    }

    if (res == ESP_OK)
    {
      size_t hlen = snprintf((char *)part_buf, 128, "\r\n--123456789000000000000987654321\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", _jpg_buf_len);
      res = httpd_resp_send_chunk(req, (const char *)part_buf, hlen);
    }
    if (res == ESP_OK)
    {
      res = httpd_resp_send_chunk(req, (const char *)_jpg_buf, _jpg_buf_len);
    }
    if (fb)
    {
      esp_camera_fb_return(fb);
      fb = NULL;
      _jpg_buf = NULL;
    }
    if (res != ESP_OK)
      break;
  }
  return res;
}

httpd_handle_t ui_httpd = NULL;     // Server สำหรับหน้าเว็บ (Port 80)
httpd_handle_t stream_httpd = NULL; // Server สำหรับวิดีโอ (Port 81)

void startWebServer()
{
  // 1. ตั้งค่า Server ตัวที่ 1 (Port 80) ไว้ส่งเว็บและ JSON
  httpd_config_t config = HTTPD_DEFAULT_CONFIG();
  config.server_port = 80;
  config.max_uri_handlers = 4;

  httpd_uri_t index_uri = {.uri = "/", .method = HTTP_GET, .handler = index_handler, .user_ctx = NULL};
  httpd_uri_t result_uri = {.uri = "/result", .method = HTTP_GET, .handler = result_handler, .user_ctx = NULL};

  if (httpd_start(&ui_httpd, &config) == ESP_OK)
  {
    httpd_register_uri_handler(ui_httpd, &index_uri);
    httpd_register_uri_handler(ui_httpd, &result_uri);
    httpd_register_err_handler(ui_httpd, HTTPD_404_NOT_FOUND, captive_portal_handler);
  }

  // 2. ตั้งค่า Server ตัวที่ 2 (Port 81) ไว้รับภาระวิดีโอสตรีมอย่างเดียว
  httpd_config_t stream_config = HTTPD_DEFAULT_CONFIG();
  stream_config.server_port = 81;
  stream_config.ctrl_port = 81; // ต้องแยกพอร์ตควบคุมด้วย
  httpd_uri_t stream_uri = {.uri = "/stream", .method = HTTP_GET, .handler = stream_handler, .user_ctx = NULL};

  if (httpd_start(&stream_httpd, &stream_config) == ESP_OK)
  {
    httpd_register_uri_handler(stream_httpd, &stream_uri);
  }
}

// ==============================================================================
// 3. AI INFERENCE LOGIC
// ==============================================================================
void setupAI()
{
  Serial.println("[AI] Allocating Tensor Arena in PSRAM...");
  tensor_arena = (uint8_t *)ps_malloc(kTensorArenaSize);
  if (!tensor_arena)
  {
    Serial.println("[AI ERROR] PSRAM Allocation Failed for Tensor Arena!");
    return;
  }

  static tflite::MicroErrorReporter micro_error_reporter;
  const tflite::Model *model = tflite::GetModel(mangosteen_quant_int8_tflite);
  if (!model)
  {
    Serial.println("[AI ERROR] Model pointer is null!");
    return;
  }

  static tflite::AllOpsResolver resolver;
  static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize, &micro_error_reporter);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk)
  {
    Serial.println("[AI ERROR] AllocateTensors failed!");
    return;
  }

  input = interpreter->input(0);
  output = interpreter->output(0);
  Serial.println("[AI OK] TensorFlow Lite Initialized Successfully.");
}

void runAIInference() {
  // 1. เช็คความพร้อมของโมเดลและกล้อง
  if (input == nullptr || output == nullptr) return;
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) return;

  // 2. ประกาศตัวแปรเตรียมข้อมูลภาพเข้าโมเดล
  int8_t* input_buffer = input->data.int8;
  float input_scale = input->params.scale;
  int input_zero_point = input->params.zero_point;
  uint16_t* rgb565_data = (uint16_t*)fb->buf;

  // 3. ลูปที่ 1: ดึงพิกเซลภาพมาแปลงสเกลสี (ทำงาน 96*96 รอบ)
  int idx = 0;
  for (int i = 0; i < 96 * 96; i++) {
    uint16_t pixel = rgb565_data[i];
    uint8_t r = ((pixel >> 11) & 0x1F) << 3;
    uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
    uint8_t b = (pixel & 0x1F) << 3;

    // แปลงเป็นสเกล -1.0 ถึง 1.0 (Symmetric) ตามที่เทรนใน Colab
    float norm_r = ((float)r - 127.5f) / 127.5f;
    float norm_g = ((float)g - 127.5f) / 127.5f;
    float norm_b = ((float)b - 127.5f) / 127.5f;

    // บีบอัดกลับเป็น int8 แล้วยัดใส่ Tensor
    input_buffer[idx++] = (int8_t)constrain(round(norm_r / input_scale) + input_zero_point, -128, 127);
    input_buffer[idx++] = (int8_t)constrain(round(norm_g / input_scale) + input_zero_point, -128, 127);
    input_buffer[idx++] = (int8_t)constrain(round(norm_b / input_scale) + input_zero_point, -128, 127);
  }

  // 4. สั่งโมเดลทำนาย
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    esp_camera_fb_return(fb);
    return;
  }

  // 5. เตรียมตัวแปรอ่านผลลัพธ์
  int8_t *out_data = output->data.int8;
  float scale = output->params.scale;
  int zero_point = output->params.zero_point;

  int best_idx = 0;
  float max_p = -999.0f;
  float probs[3];

  // 6. ลูปที่ 2: คำนวณเปอร์เซ็นต์ทั้ง 3 คลาส (ทำงาน 3 รอบ)
  for (int i = 0; i < 3; i++) {
    probs[i] = (out_data[i] - zero_point) * scale;
    if (probs[i] > max_p) {
      max_p = probs[i];
      best_idx = i;
    }
  }

  // 7. นำผลลัพธ์บันทึกไว้สำหรับส่งให้หน้าเว็บ
  strncpy(latest_result.best_class, class_names[best_idx], sizeof(latest_result.best_class) - 1);
  latest_result.probabilities[0] = probs[0]; // overripe
  latest_result.probabilities[1] = probs[1]; // ripe
  latest_result.probabilities[2] = probs[2]; // unripe

  esp_camera_fb_return(fb);
}

// ==============================================================================
// 4. SETUP & MAIN LOOP
// ==============================================================================
const byte DNS_PORT = 53;
DNSServer dnsServer;
bool wifi_started = false;

void setup()
{
  Serial.begin(115200);
  delay(1000);

  pinMode(PWR_ON_PIN, OUTPUT);
  digitalWrite(PWR_ON_PIN, HIGH);
  delay(150);

  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = CAM_Y2_PIN;
  config.pin_d1 = CAM_Y3_PIN;
  config.pin_d2 = CAM_Y4_PIN;
  config.pin_d3 = CAM_Y5_PIN;
  config.pin_d4 = CAM_Y6_PIN;
  config.pin_d5 = CAM_Y7_PIN;
  config.pin_d6 = CAM_Y8_PIN;
  config.pin_d7 = CAM_Y9_PIN;
  config.pin_xclk = CAM_XCLK_PIN;
  config.pin_pclk = CAM_PCLK_PIN;
  config.pin_vsync = CAM_VSYNC_PIN;
  config.pin_href = CAM_HREF_PIN;
  config.pin_sccb_sda = CAM_SIOD_PIN;
  config.pin_sccb_scl = CAM_SIOC_PIN;
  config.pin_pwdn = CAM_PWDN_PIN;
  config.pin_reset = CAM_RESET_PIN;

  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA;
  config.jpeg_quality = 12;
  config.fb_count = 2;
  config.fb_location = CAMERA_FB_IN_PSRAM;

  if (esp_camera_init(&config) != ESP_OK)
  {
    while (true)
    {
      delay(1000);
    }
  }

  setupAI();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD);

  dnsServer.start(DNS_PORT, "*", WiFi.softAPIP());
  wifi_started = true;

  startWebServer();
}

void loop()
{
  if (wifi_started)
  {
    dnsServer.processNextRequest();
  }

  runAIInference();
  delay(2000);
}