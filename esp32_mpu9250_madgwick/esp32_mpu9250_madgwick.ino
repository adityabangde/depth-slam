#include <Wire.h>
#include <MPU9250.h>

MPU9250 imu;

// ===== User settings =====
const uint8_t SDA_PIN = 21;
const uint8_t SCL_PIN = 22;
const uint8_t IMU_ADDR = 0x68;

const uint16_t SAMPLE_HZ = 100;      // output rate
const int AVG_WIN = 5;              // moving average window for yaw
const bool AUTO_ZERO_AT_START = true;

// ===== Circular moving average buffer =====
float yawBuf[AVG_WIN];
int yawIdx = 0;
bool yawFull = false;

float yawRef = 0.0f;
bool yawZeroed = false;

// Push angle into circular buffer
void yawPush(float deg) {
  yawBuf[yawIdx] = deg;
  yawIdx = (yawIdx + 1) % AVG_WIN;
  if (yawIdx == 0) yawFull = true;
}

// Circular average for angles in degrees
float yawAverage() {
  int n = yawFull ? AVG_WIN : yawIdx;
  if (n <= 0) return 0.0f;

  float s = 0.0f;
  float c = 0.0f;

  for (int i = 0; i < n; i++) {
    float r = yawBuf[i] * DEG_TO_RAD;
    s += sinf(r);
    c += cosf(r);
  }

  return atan2f(s, c) * RAD_TO_DEG;
}

// Normalize angle to [-180, 180]
float wrap180(float a) {
  while (a > 180.0f) a -= 360.0f;
  while (a < -180.0f) a += 360.0f;
  return a;
}

void calibrateIMU() {
  Serial.println("# Keep sensor flat and still");
  Serial.println("# Auto calibrating accel + gyro in 3 seconds...");
  delay(3000);

  imu.calibrateAccelGyro();

  Serial.println("# Now rotate sensor slowly in a figure-8");
  Serial.println("# Auto calibrating magnetometer in 10 seconds...");
  delay(1000);
  imu.calibrateMag();

  Serial.println("# Calibration complete");
}

void setup() {
  Serial.begin(115200);
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  delay(200);

  if (!imu.setup(IMU_ADDR)) {
    Serial.println("ERR: MPU9250 not found. Check wiring and address.");
    while (1) delay(500);
  }

  calibrateIMU();

  // Prime the filter / sensor output a bit
  Serial.println("# Settling...");
  for (int i = 0; i < 100; i++) {
    imu.update();
    delay(10);
  }

  // Optional zero reference from the first valid yaw
  if (AUTO_ZERO_AT_START) {
    imu.update();
    yawRef = imu.getYaw();
    yawZeroed = true;
  }

  Serial.println("# Streaming smooth yaw in degrees");
  Serial.println("# Commands:");
  Serial.println("#   z  -> re-zero yaw");
  Serial.println("#   c  -> recalibrate all");
}

void loop() {
  static uint32_t lastMs = 0;
  uint32_t now = millis();

  if (now - lastMs < (1000 / SAMPLE_HZ)) return;
  lastMs = now;

  if (!imu.update()) return;

  float yaw = imu.getYaw();

  if (!yawZeroed) {
    yawRef = yaw;
    yawZeroed = true;
  }

  float yawRel = wrap180(yaw - yawRef);

  // Smooth with circular moving average
  yawPush(yawRel);
  float yawSmooth = wrap180(yawAverage());

  Serial.println(yawSmooth, 3);

  // Serial commands
  while (Serial.available()) {
    char c = Serial.read();

    if (c == 'z' || c == 'Z') {
      yawRef = yaw;
      yawZeroed = true;

      // reset average buffer so old values do not pollute the new zero
      yawIdx = 0;
      yawFull = false;

      Serial.println("# yaw zeroed");
    }

    if (c == 'c' || c == 'C') {
      yawIdx = 0;
      yawFull = false;
      yawZeroed = false;

      calibrateIMU();

      // settle again
      for (int i = 0; i < 100; i++) {
        imu.update();
        delay(10);
      }

      imu.update();
      yawRef = imu.getYaw();
      yawZeroed = true;
      Serial.println("# recalibration done");
    }
  }
}
