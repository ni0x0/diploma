#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>

#include <WiFi.h>

#include <AccelStepper.h>

#include <Preferences.h>

// Выводы ESP32
#define MENU_BUTTON_PIN 18    // Пин для кнопки навигации по меню
#define ACTION_BUTTON_PIN 19  // Пин для кнопки выбора/действия
#define POTENCIOMETER_PIN 35  // Пин для считывания значения потенциометра

#define POWER_LED_PIN 33  // Пин для индикации питания (LED)
#define MOTOR_LED_PIN 32  // Пин для индикации работы мотора (LED)

#define DRV_STEP_PIN 26  // Пин STEP драйвера шагового мотора
#define DRV_DIR_PIN 25   // Пин DIR драйвера шагового мотора
#define DRV_M0_PIN 12
#define DRV_M1_PIN 14
#define DRV_M2_PIN 27
#define DRV_EN_PIN 13

#define DEBOUNCE_DELAY 20  // Период антидребезга для кнопок (в мс)

#define SCREEN_WIDTH 128  // Ширина OLED-дисплея в пикселях
#define SCREEN_HEIGHT 64  // Высота OLED-дисплея в пикселях

#define SERVER_PORT 1234  // Порт TCP-сервера для соединений

#define WAIT_TIME 1000

#define POT_MIN_VALUE 0     // Минимальное значение потенциометра
#define POT_MAX_VALUE 4095  // Максимальное

#define MOTOR_STEPS 200                                 // Количество шагов для полного оборота мотора
#define MICROSTEPS 8                                    // Деление шага на 8
#define FULL_ROTATION_STEPS (MOTOR_STEPS * MICROSTEPS)  // Количество шагов для полного оборота с учетом деления шага
#define GEAR_DIV 3                                      // Соотношение передачи шестерен

#define MOTOR_BASE_MIN_SPEED 0    // Минимальная скорость мотора
#define MOTOR_BASE_MAX_SPEED 300  // Максимальная скорость мотора
#define MOTOR_BASE_DEF_ACCEL 60   // Ускорение по умолчанию

// С учетом микрошагов:
#define MOTOR_MICRO_MIN_SPEED (MOTOR_BASE_MIN_SPEED * MICROSTEPS)  // Мин ск-ть -  0
#define MOTOR_MICRO_MAX_SPEED (MOTOR_BASE_MAX_SPEED * MICROSTEPS)  // Макс ск-ть - 2400
#define MOTOR_MICRO_DEF_ACCEL (MOTOR_BASE_DEF_ACCEL * MICROSTEPS)  // Ускорение -  480

// Переменные для антидребезга кнопки действия
bool action_button_last_state = LOW;                 // Предыдущее состояние кнопки действия
bool action_button_state = LOW;                      // Текущее состояние кнопки действия
unsigned long action_button_last_debounce_time = 0;  // Время последнего изменения состояния кнопки действия

// Переменные для антидребезга кнопки меню
bool menu_button_last_state = LOW;                 // Предыдущее состояние кнопки меню
bool menu_button_state = LOW;                      // Текущее состояние кнопки меню
unsigned long menu_button_last_debounce_time = 0;  // Время последнего изменения состояния кнопки меню

// Инициализация переменных для индикации состояния мотора
bool motor_led_state = LOW;  // Текущее состояние LED-индикатора мотора

// Объект для работы с OLED-дисплеем (Adafruit SSD1306)
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

// Объект для управления шаговым двигателем через библиотеку AccelStepper
AccelStepper stepper(AccelStepper::DRIVER, DRV_STEP_PIN, DRV_DIR_PIN);

// TCP-сервер и клиент для обмена данными по Wi-Fi
WiFiServer server(SERVER_PORT);
WiFiClient activeClient;

// Настройки Wi-Fi подключения
const char* ap_ssid = "ESP32-AP";      // SSID точки доступа
const char* ap_password = "12345678";  // Пароль точки доступа
char sta_ssid[32] = "Xiaom";           //{ 0 };    // SSID домашней сети (режим STA)
char sta_password[32] = "plmNko553";   //{ 0 };    // Пароль домашней сети

bool currentModeIsAP = true;   // Текущий режим: true — AP, false — STA
bool clientConnected = false;  // Есть ли подключённый клиент по TCP

// Состояния меню
enum DeviceMode {
  MOTOR_TEST,       // Режим теста мотора
  WIFI_CONFIG,      // Режим настройки подключения
  SCAN_MODE,        // Режим сканирования
  DEVICE_INFO,      // Информация об устройстве
  MENU_ITEMS_NUM,   // Общее количество пунктов меню
  MAIN_MENU_OUTPUT  // Отображение главного меню
};

// Названия пунктов меню
const char* menuItems[] = {
  "Motor test",
  "Connection Config",
  "Scanning mode",
  "Device Info"
};

// Выбранный пункт меню (потенциометром)
uint8_t potenciometer_selected_item = 0;

// Текущий активный режим устройства
DeviceMode current_device_mode = MAIN_MENU_OUTPUT;

// ====== MOTOR TEST MODE ======
bool test_mode_motor_is_running = false;  // Запущен ли мотор в режиме теста
int motor_test_speed = 0;                 // Скорость мотора в режиме теста

// ====== SCAN MODE ======
bool scanning_outp_is_position = true;   // Тип выводимой информации о сканировании: через число позиций или через угол
int motor_scan_speed = 0;                // Скорость мотора
int motor_scan_acceleration = 0;         // Ускорение мотора
bool scan_in_progress = false;           // Ведётся ли в данный момент сканирование
int scan_number_of_turns_to_do = 0;      // Сколько поворотов нужно сделать
int scan_number_of_turns_completed = 0;  // Сколько уже сделано
int scan_turn_delta = 0;                 // Угол (в шагах), на который поворачивается мотор за раз
bool scan_abort_request = false;         // Наличие запроса на прерывание сканирования
bool scan_start_request = false;         // Наличие запроса старт сканирования
bool scan_continue_request = false;      // Наличие запроса на продолжение сканирования

#define PACKET_SIZE 65
#define RESPONSE_SIZE 3

uint8_t rx_buffer[PACKET_SIZE];
uint8_t tx_response[RESPONSE_SIZE];

// Семафор и объект мьютекса для синхронизации доступа к общим данным между задачами
SemaphoreHandle_t xMutex;
portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;

// Смещение выводимой информации на дисплее
int VERTICAL_OFFSET = 4;

void setup() {

  Serial.begin(115200);
  delay(100);

  // Установка режимов выводов
  pinMode(MENU_BUTTON_PIN, INPUT);
  pinMode(ACTION_BUTTON_PIN, INPUT);
  pinMode(POTENCIOMETER_PIN, INPUT);

  pinMode(POWER_LED_PIN, OUTPUT);
  pinMode(MOTOR_LED_PIN, OUTPUT);

  pinMode(DRV_EN_PIN, OUTPUT);
  pinMode(DRV_M0_PIN, OUTPUT);
  pinMode(DRV_M1_PIN, OUTPUT);
  pinMode(DRV_M2_PIN, OUTPUT);

  digitalWrite(DRV_EN_PIN, HIGH);  // Выключение шаговика
  digitalWrite(DRV_M0_PIN, HIGH);  // Установка 1/8 шага: M0=HIGH, M1=HIGH, M2=LOW
  digitalWrite(DRV_M1_PIN, HIGH);
  digitalWrite(DRV_M2_PIN, LOW);

  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    Serial.println(F("SSD1306 allocation failed"));
    while (true);
  }

  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(WHITE);
  //drawMenu(selectedItem); // нет в full

  // Инициализация семафора
  xMutex = xSemaphoreCreateMutex();

  // loadWiFiSettings();

  // Запуск задач
  xTaskCreate(
    buttonTask,    // Функция задачи
    "ButtonTask",  // Имя задачи
    2048,          // Размер стека
    NULL,          // Параметры
    1,             // Приоритет
    NULL           // Дескриптор задачи
  );

  xTaskCreate(
    serverTask,
    "ServerTask",
    4096,
    NULL,
    2,
    NULL
  );

  // Начальная инициализация в режиме AP
  initAPMode();

  Serial.println("Initialization is completed");
  digitalWrite(POWER_LED_PIN, HIGH);
}

void loop() {

  DeviceMode current_device_mode_copy;

  xSemaphoreTake(xMutex, portMAX_DELAY);
  current_device_mode_copy = current_device_mode;
  xSemaphoreGive(xMutex);

  switch (current_device_mode_copy) {
    case MAIN_MENU_OUTPUT:
      Serial.println("Entering Menu Output mode");
      output_main_menu();
      break;
    case MOTOR_TEST:
      Serial.println("Entering Motor Test mode");
      motor_test();
      break;
    case WIFI_CONFIG:
      Serial.println("Entering Wifi Config mode");
      connection_outp();
      break;
    case SCAN_MODE:
      Serial.println("Entering Scanning mode");
      scanning_process();
      break;
    case DEVICE_INFO:
      Serial.println("Entering Info Output mode");
      info_outp();
      break;
    default:
      break;
  }
}

// Задача обработки подключения и сообщений от клиента
void serverTask(void *pvParameters) {
  while (1) {
    xSemaphoreTake(xMutex, portMAX_DELAY);
    if (serverRunning()) {
      handleClientConnections();
    }
    xSemaphoreGive(xMutex);

    vTaskDelay(pdMS_TO_TICKS(100));
  }
}

// Задача обработки нажатия кнопок
void buttonTask(void* pvParameters) {

  while (1) {
    // Чтение текущего состояния кнопок
    bool menu_reading = digitalRead(MENU_BUTTON_PIN);
    bool action_reading = digitalRead(ACTION_BUTTON_PIN);

    // Обработка кнопки меню
    if (menu_reading != menu_button_last_state) {
      menu_button_last_debounce_time = millis();
    }

    if ((millis() - menu_button_last_debounce_time) > DEBOUNCE_DELAY) {
      if (menu_reading != menu_button_state) {
        menu_button_state = menu_reading;

        if (menu_button_state == LOW) {
          toggle_menu_button();
        }
      }
    }

    menu_button_last_state = menu_reading;

    // Обработка кнопки действия
    if (action_reading != action_button_last_state) {
      action_button_last_debounce_time = millis();
    }

    if ((millis() - action_button_last_debounce_time) > DEBOUNCE_DELAY) {
      if (action_reading != action_button_state) {
        action_button_state = action_reading;

        if (action_button_state == LOW) {
          toggle_action_button();
        }
      }
    }

    action_button_last_state = action_reading;

    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

// Функция переключения меню
void toggle_menu_button() {
  //taskENTER_CRITICAL(&mux);
  xSemaphoreTake(xMutex, portMAX_DELAY);
  
  Serial.println("Menu button click processing");

  if (current_device_mode != MAIN_MENU_OUTPUT) {
    current_device_mode = MAIN_MENU_OUTPUT;
  } else {
    // Ограничение индекса выбранного пункта допустимыми значениями
    if (potenciometer_selected_item >= MENU_ITEMS_NUM) {
      potenciometer_selected_item = 0;
    }

    current_device_mode = static_cast<DeviceMode>(potenciometer_selected_item);
  }

  //taskEXIT_CRITICAL(&mux);
  xSemaphoreGive(xMutex);
}

// Функция обработки действия
void toggle_action_button() {
  //taskENTER_CRITICAL(&mux);
  xSemaphoreTake(xMutex, portMAX_DELAY);

  Serial.println("Action button click processing");

  switch (current_device_mode) {
    case MOTOR_TEST:
      test_mode_motor_is_running = !test_mode_motor_is_running;
      
      digitalWrite(MOTOR_LED_PIN, test_mode_motor_is_running);
      digitalWrite(DRV_EN_PIN, test_mode_motor_is_running ? LOW : HIGH);

      Serial.print("Test mode: motor is running - ");
      Serial.println(test_mode_motor_is_running);
      break;
    case SCAN_MODE:
      scanning_outp_is_position = !scanning_outp_is_position;
      break;
    case WIFI_CONFIG:
      toggleWiFiMode();
      break;
    default:
      break;
  }

  //taskEXIT_CRITICAL(&mux);
  xSemaphoreGive(xMutex);
}

// Инициализация режима АР
void initAPMode() {
  //xSemaphoreTake(xMutex, portMAX_DELAY);

  Serial.println("Инициализация АР режима");

  WiFi.disconnect(true);
  WiFi.mode(WIFI_AP);

  // WiFi.softAP(ap_ssid, ap_password);

  // Проверяем, удалось ли запустить AP
  if (!WiFi.softAP(ap_ssid, ap_password)) {
    Serial.println("!!Ошибка запуска AP!!");
  }
  else {
    Serial.println("AP mode is activated");
    Serial.print("AP IP: ");
    Serial.println(WiFi.softAPIP());

    server.begin();
    currentModeIsAP = true;
  }

  //xSemaphoreGive(xMutex);
}

// Инициализация режима STA
void initSTAMode() {
  //xSemaphoreTake(xMutex, portMAX_DELAY);

  Serial.println("Инициализация STA режима");

  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_STA);
  WiFi.begin(sta_ssid, sta_password);

  Serial.println("Подключение к Wi-Fi...");

  unsigned long startTime = millis();
  while (WiFi.status() != WL_CONNECTED && millis() - startTime < 60000) {
    vTaskDelay(pdMS_TO_TICKS(500));
    Serial.print(".");
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nПодключено!");
    Serial.print("STA IP: ");
    Serial.println(WiFi.localIP());
    currentModeIsAP = false;
    server.begin();
  } else {
    Serial.println("\nОшибка подключения! Возврат в AP режим");
    //initAPMode();
  }

  //xSemaphoreGive(xMutex);
}

// Сменить режим сервера
void toggleWiFiMode() {
  //xSemaphoreTake(xMutex, portMAX_DELAY);

  Serial.println("Toggle Wifi Mode");

  if (clientConnected) {
    // activeClient.println("Сервер меняет режим работы"); менять !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    activeClient.stop();
    clientConnected = false;
  }

  server.stop();

  if (currentModeIsAP) {
    initSTAMode();
  } else {
    initAPMode();
  }

  //xSemaphoreGive(xMutex);
}

bool serverRunning() {
  return WiFi.status() == WL_CONNECTED || currentModeIsAP;
}

// // int bytesToInt(const uint8_t* data) {
// //   // Считаем сумму 8 двухбайтовых значений
// //   int result = 0;
// //   for (int i = 0; i < 16; i += 2) {
// //     result += (data[i] | (data[i + 1] << 8));
// //   }
// //   return result;
// // }

  // static String command = "";
  
  // // if (activeClient.available()) {
  // //   char c = activeClient.read();
  // //   if (c == '\n') {
  // //     command.trim();
  // //     Serial.println("Получена команда: " + command);


  // //     command = "";  // сбрасываем
  // //   } else {
  // //     command += c;
  // //   }
  // // }

uint16_t bytesToUint16(const uint8_t* ptr) {
  return (uint16_t(ptr[0]) | (uint16_t(ptr[1]) << 8));
}

void sendResponse(uint8_t code, uint16_t data = 0) {
  tx_response[0] = code;
  tx_response[1] = data & 0xFF;
  tx_response[2] = (data >> 8) & 0xFF;
  activeClient.write(tx_response, RESPONSE_SIZE);
}

// Непосредственная работа с клиентом/ами
// Семафоры не ставить. Они выше
void handleClientConnections() {
  // Проверка подключения клиента
  if (!activeClient || !activeClient.connected()) {
    if (clientConnected) {
      Serial.println("Клиент отключен");
      activeClient.stop();
      clientConnected = false;
    }

    activeClient = server.available();
    if (activeClient) {
      clientConnected = true;
      Serial.println("Новый клиент подключен");
      //activeClient.println("Добро пожаловать");
    }
    return;
  }

  // if (activeClient.available() >= PACKET_SIZE) {
  while (activeClient.available() >= PACKET_SIZE) {
    activeClient.readBytes(rx_buffer, PACKET_SIZE);

    uint8_t cmd = rx_buffer[0];

    switch (cmd) {
      case 0x01: {  // Начать сканирование
        uint16_t turns       = bytesToUint16(rx_buffer + 1);
        uint16_t angle       = bytesToUint16(rx_buffer + 17);
        uint16_t speed       = bytesToUint16(rx_buffer + 33);
        uint16_t acceleration= bytesToUint16(rx_buffer + 49);

        motor_scan_speed = speed;
        motor_scan_acceleration = acceleration;
        scan_turn_delta = angle;
        scan_number_of_turns_to_do = turns;
        scan_start_request = true;

        Serial.printf("Получили команду начать сканирование\n");
        Serial.printf("Поворотов: %u, Угол: %u, Скорость: %u, Ускорение: %u\n",
                      scan_number_of_turns_to_do, scan_turn_delta, motor_scan_speed, motor_scan_acceleration);
        break;
      }

      case 0x02: {  // Продолжить сканирование
        scan_continue_request = true;
        Serial.println("Получил команду продолжить сканирование");
        break;
      }

      case 0x03: {  // Прервать сканирование
        scan_abort_request = true;

        Serial.println("Команда: прервать сканирование");
        // sendResponse(0x02, 0);  // подтверждение остановки
        break;
      }

      case 0x04: {  // Установка Wi-Fi
        char ssid[33] = {0};
        char pass[33] = {0};

        memcpy(ssid, rx_buffer + 1, 32);
        memcpy(pass, rx_buffer + 33, 32);

        Serial.printf("Команда: установить параметры Wi-Fi\n");
        Serial.printf("SSID: %s\n", ssid);
        Serial.printf("Password: %s\n", pass);

        // change_sta_settings(ssid, pass); 

        // sendResponse(0x03, 0);  // подтверждение сохранения
        break;
      }

      case 0x05: {  // Сброс подключения
        Serial.println("Команда: сброс подключения");
        // sendResponse(0x02, 0);  // подтверждение
        // activeClient.stop();
        // clientConnected = false;
        break;
      }

      default: {
        Serial.printf("Неизвестная команда: 0x%02X\n", cmd);
        break;
      }
    }
  }
}

  // while (activeClient.available()) {  // <-- Используем while, а не if, чтобы читать все доступные байты
  //   char c = activeClient.read();
  //   if (c == '\n') {
  //     command.trim();
  //     if (command.length() > 0) {  // Выводим только непустые команды
  //       Serial.println("Получена команда: " + command);
  //       // Здесь можно добавить обработку команды (например, ответ клиенту)
  //       activeClient.println("ESP получил: " + command);  // Отправляем подтверждение
  //     }
  //     command = "";  // Сбрасываем буфер
  //   } else if (c != '\r') {  // Игнорируем символ возврата каретки (\r)
  //     command += c;
  //     if (command.length() > 64) {  // Защита от переполнения
  //       command = "";  // Сбрасываем, если команда слишком длинная
  //       Serial.println("Ошибка: слишком длинная команда");
  //     }
  //   }
  // }

  // // Если есть подключение и достаточно данных
  // // if (activeClient.available() >= 65) {
  // //   uint8_t buffer[65];
  // //   activeClient.readBytes(buffer, 65);

  // //   uint8_t command = buffer[0];

  // //   switch (command) {
  // //     case 0x01: { // Начать сканирование
  // //       taskENTER_CRITICAL(&mux);
  // //       scan_number_of_turns_to_do = bytesToInt(buffer + 1);
  // //       scan_turn_delta = bytesToInt(buffer + 17);
  // //       motor_scan_speed = bytesToInt(buffer + 33);
  // //       motor_scan_acceleration = bytesToInt(buffer + 49);
  // //       scan_start_request = true;
  // //       taskEXIT_CRITICAL(&mux);
  // //       Serial.println("Команда: начать сканирование");
  // //       break;
  // //     }

  // //     case 0x02: // Продолжить сканирование
  // //       scan_continue_request = true;
  // //       Serial.println("Команда: продолжить сканирование");
  // //       break;

  // //     case 0x03: // Прервать сканирование
  // //       scan_abort_request = true;
  // //       Serial.println("Команда: прервать сканирование");
  // //       break;

  // //     case 0x04: { // Установка параметров Wi-Fi
  // //       memcpy(sta_ssid, buffer + 1, 32);
  // //       memcpy(sta_password, buffer + 33, 32);
  // //       sta_ssid[31] = '\0';
  // //       sta_password[31] = '\0';
  // //       change_sta_settings();
  // //       Serial.println("Команда: установить параметры Wi-Fi");
  // //       break;
  // //     }

  // //     case 0x05: { // Сброс подключения
  // //       Serial.println("Команда: сброс подключения");
  // //       activeClient.stop();
  // //       clientConnected = false;
  // //       break;
  // //     }

  // //     default:
  // //       Serial.print("Неизвестная команда: ");
  // //       Serial.println(command, HEX);
  // //       break;
  // //   }
  // // }

// // Изменить WiFi настройки
// void change_sta_settings() {
//   Preferences prefs;
//   prefs.begin("wifi", false); // "wifi" — пространство имён

//   // Сохраняем SSID и пароль
//   prefs.putString("ssid", sta_ssid);
//   prefs.putString("pass", sta_password);
//   prefs.end();

//   Serial.println("Wi-Fi настройки сохранены во flash");

//   // Отправка клиенту: команда подтверждения (3)
//   uint8_t response[3] = {3, 0, 0};
//   if (activeClient && activeClient.connected()) {
//     activeClient.write(response, 3);
//   }

//   // Переподключение, если мы не в AP-режиме
//   if (!currentModeIsAP) {
//     initSTAMode();
//   }
// }

// // Загрузка настроек STA
// void loadWiFiSettings() {
//   Preferences prefs;
//   prefs.begin("wifi", true); // true — только чтение

//   String ssid = prefs.getString("ssid", "");
//   String pass = prefs.getString("pass", "");

//   ssid.toCharArray(sta_ssid, sizeof(sta_ssid));
//   pass.toCharArray(sta_password, sizeof(sta_password));

//   prefs.end();
// }

void drawBoldText(int x, int y, const char* text) {
  display.setCursor(x, y);
  display.println(text);
  display.setCursor(x + 1, y);
  display.println(text);
}

void drawThinRect(int x, int y, int w, int h) {
  display.drawRect(x, y, w, h, WHITE);
}

// Меню вывода главного меню
void output_main_menu() {
  int previousSelectedItem = -1;

  //while (current_device_mode == MAIN_MENU_OUTPUT) {
  DeviceMode current_device_mode_copy;
  while (true) {

    xSemaphoreTake(xMutex, portMAX_DELAY);
    current_device_mode_copy = current_device_mode;
    xSemaphoreGive(xMutex);

    if (current_device_mode_copy != MAIN_MENU_OUTPUT) {
      break;
    }

    int analogValue = analogRead(POTENCIOMETER_PIN);
    int selectedItem = map(analogValue, POT_MIN_VALUE, POT_MAX_VALUE, MENU_ITEMS_NUM - 1, 0);

    // Если выбранный пункт изменился — перерисовываем меню
    if (selectedItem != previousSelectedItem) {
      Serial.print("New Item: ");
      Serial.println(menuItems[selectedItem]);

      previousSelectedItem = selectedItem;

      xSemaphoreTake(xMutex, portMAX_DELAY);
      potenciometer_selected_item = selectedItem;
      xSemaphoreGive(xMutex);

      display.clearDisplay();

      // Заголовок
      drawBoldText(0, 0 + VERTICAL_OFFSET, "Main Menu");

      // Пункты меню
      for (int i = 0; i < MENU_ITEMS_NUM; i++) {
        int y = 14 + i * 12 + VERTICAL_OFFSET;

        if (i == selectedItem) {
          drawThinRect(0, y - 2, SCREEN_WIDTH - 1, 12);
          drawBoldText(2, y, menuItems[i]);
        } else {
          display.setCursor(2, y);
          display.println(menuItems[i]);
        }
      }

      display.display();
    }

    vTaskDelay(pdMS_TO_TICKS(50));
  }
}

// Тест мотора
void motor_test() {
  const int POT_CENTER = POT_MAX_VALUE / 2;
  const int SPEED_LEVELS = 36;              // Количество уровней скорости (дискретизация)
  const int DISPLAY_UPDATE_INTERVAL = 200;  // Обновлять дисплей раз в 100 мс

  // Отрисовка экрана
  display.clearDisplay();
  drawBoldText(0, 0 + VERTICAL_OFFSET, "Motor test");
  display.setCursor(0, 16 + VERTICAL_OFFSET);
  display.println("Rotation speed:");
  display.display();

  // Настройка мотора
  xSemaphoreTake(xMutex, portMAX_DELAY);
  digitalWrite(MOTOR_LED_PIN, test_mode_motor_is_running);
  digitalWrite(DRV_EN_PIN, test_mode_motor_is_running ? LOW : HIGH);
  xSemaphoreGive(xMutex);

  stepper.setMaxSpeed(MOTOR_MICRO_MAX_SPEED);      // Максимальная скорость (шагов в секунду)
  stepper.setAcceleration(MOTOR_MICRO_DEF_ACCEL);  // Ускорение

  // Для троттлинга обновления дисплея
  uint32_t last_display_update = -1;
  int last_displayed_speed = -1;

  DeviceMode current_device_mode_copy;
  bool test_mode_motor_is_running_copy;

  //while (current_device_mode == MOTOR_TEST) {
  while (true) {

    xSemaphoreTake(xMutex, portMAX_DELAY);
    current_device_mode_copy = current_device_mode;
    test_mode_motor_is_running_copy = test_mode_motor_is_running;
    xSemaphoreGive(xMutex);

    if (current_device_mode_copy != MOTOR_TEST) {
      break;
    }

    int potValue = analogRead(POTENCIOMETER_PIN);
    int delta = POT_CENTER - potValue;
    int stepSpeed = 0;
    int stepSpeed_run = 0;
    
    // Преобразуем в диапазон -MAX_SPEED … +MAX_SPEED
    delta = constrain(delta, -POT_CENTER, POT_CENTER);
    int speed_level = map(abs(delta), 0, POT_CENTER, 0, SPEED_LEVELS);
    stepSpeed = (delta > 0 ? 1 : -1) * map(speed_level, 0, SPEED_LEVELS, 0, MOTOR_MICRO_MAX_SPEED);

    if (test_mode_motor_is_running_copy) {  
      stepSpeed_run = stepSpeed;
    } else {
      stepSpeed_run = 0;
    }

    stepper.setSpeed(stepSpeed_run);
    stepper.runSpeed();

    // Расчёт скорости в градусах/сек
    float deg_per_sec = (float)stepSpeed * 360.0 / (MOTOR_STEPS * MICROSTEPS * GEAR_DIV);
    int rounded_deg = round(deg_per_sec);

    // Обновляем дисплей, только если: 1) Прошло > 200 мс И 2) Скорость изменилась
    if (millis() - last_display_update >= DISPLAY_UPDATE_INTERVAL && rounded_deg != last_displayed_speed) {
      char buffer[20];
      snprintf(buffer, sizeof(buffer), " %d degree/sec", rounded_deg);  //.1f // разделить на 3 для платформы

      int textWidth = 6 * strlen(buffer);
      int xCentered = (SCREEN_WIDTH - textWidth) / 2;

      int y = 32 + VERTICAL_OFFSET;

      display.fillRect(0, y, SCREEN_WIDTH, 10, BLACK);  // Типо очищаем строку
      drawBoldText(xCentered, y, buffer);
      display.display();

      last_display_update = millis();
      last_displayed_speed = rounded_deg;
    }

    vTaskDelay(1);
  }

  stepper.setSpeed(0);
  digitalWrite(MOTOR_LED_PIN, LOW);
  digitalWrite(DRV_EN_PIN, HIGH);
}

void connection_outp() {
  display.clearDisplay();

  bool prevModeIsAP = !currentModeIsAP;
  IPAddress prevIP(1, 1, 1, 1);
  bool prevClientConnected = !clientConnected;

  DeviceMode current_device_mode_copy;

  while (true) {

    xSemaphoreTake(xMutex, portMAX_DELAY);
    current_device_mode_copy = current_device_mode;    
    bool wifiModeChanged = (currentModeIsAP != prevModeIsAP);

    IPAddress currentIP;
    //IPAddress currentIP = currentModeIsAP ? WiFi.softAPIP() : WiFi.localIP();
    if (currentModeIsAP) {
      currentIP = WiFi.softAPIP();
    } else {
      currentIP = (WiFi.status() == WL_CONNECTED) ? WiFi.localIP() : INADDR_NONE;
    }

    bool ipChanged = (currentIP != prevIP);
    bool clientChanged = (clientConnected != prevClientConnected);
    xSemaphoreGive(xMutex);

    if (current_device_mode_copy != WIFI_CONFIG) {
      break;
    }

    if (wifiModeChanged || ipChanged || clientChanged) {

      Serial.println("Обновление вывода в wifi config");

      // Обновляем предыдущее состояние
      prevModeIsAP = currentModeIsAP;
      prevIP = currentIP;
      prevClientConnected = clientConnected;

      // Строка режима подключения
      const char* modeStr = currentModeIsAP ? "AP" : "STA";

      // IP строка
      char ipStr[20];
      if (currentIP == INADDR_NONE || currentIP[0] == 0) {
        strcpy(ipStr, "X.X.X.X");
      } else {
        snprintf(ipStr, sizeof(ipStr), "%d.%d.%d.%d", currentIP[0], currentIP[1], currentIP[2], currentIP[3]);
      }

      // Статус клиента
      const char* clientStr = clientConnected ? "Connected" : "Not connected";

      // Перерисовка экрана
      display.clearDisplay();
      drawBoldText(0, 0 + VERTICAL_OFFSET, "WiFi config");

      display.setCursor(0, 16 + VERTICAL_OFFSET);
      display.print("Server Conn Mode: ");
      display.println(modeStr);

      display.setCursor(0, 28 + VERTICAL_OFFSET);
      display.print("IP: ");
      display.println(ipStr);

      display.setCursor(0, 40 + VERTICAL_OFFSET);
      display.print("Client: ");
      display.println(clientStr);

      display.display();
    }

    vTaskDelay(pdMS_TO_TICKS(500));
  }
}

// bool scanning_outp_is_position = true;   // Тип выводимой информации о сканировании: через число позиций или через угол
// int motor_scan_speed = 0;                // Скорость мотора
// int motor_scan_acceleration = 0;         // Ускорение мотора
// bool scan_in_progress = false;           // Ведётся ли в данный момент сканирование
// int scan_number_of_turns_to_do = 0;      // Сколько поворотов нужно сделать
// int scan_number_of_turns_completed = 0;  // Сколько уже сделано
// int scan_turn_delta = 0;                 // Угол (в шагах), на который поворачивается мотор за раз
// bool scan_abort_request = false;         // Наличие запроса на прерывание сканирования
// bool scan_start_request = false;         // Наличие запроса старт сканирования
// bool scan_continue_request = false;      // Наличие запроса на продолжение сканирования

// Режим сканирования
void scanning_process() {
  bool prev_scan_outp = !scanning_outp_is_position;
  bool scan_outp_changed = false;

  scan_number_of_turns_completed = 0;
  scan_number_of_turns_to_do = 0;

  scan_abort_request = false;
  scan_start_request = false;        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  scan_in_progress = false;
  scan_continue_request = false;

  DeviceMode current_device_mode_copy;

  while (1) {

    xSemaphoreTake(xMutex, portMAX_DELAY);
    current_device_mode_copy = current_device_mode;  

    scan_outp_changed = (prev_scan_outp != scanning_outp_is_position);
    xSemaphoreGive(xMutex);

    if (current_device_mode_copy != SCAN_MODE) {
      break;
    }

    if (scan_outp_changed) {

      Serial.println("Обновление вывода в scan config");
      prev_scan_outp = scanning_outp_is_position;

      drawScanModeScreen(prev_scan_outp, false, 11, 22);
    }

    if (scan_start_request) {
      Serial.println("Обрабатываю запрос на начало сканирования");

      float stepSpeed = motor_scan_speed * (MOTOR_STEPS * MICROSTEPS * GEAR_DIV) / 360.0;
      float accelStep = motor_scan_acceleration * (MOTOR_STEPS * MICROSTEPS * GEAR_DIV) / 360.0;

      Serial.print("Преобразование скорости: ");
      Serial.print(motor_scan_speed);
      Serial.print(" -> ");
      Serial.println(stepSpeed);

      Serial.print("Преобразование ускорения: ");
      Serial.print(motor_scan_acceleration);
      Serial.print(" -> ");
      Serial.println(accelStep);

      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      int32_t delta_steps = 117; (int32_t)roundf(scan_turn_delta * (MOTOR_STEPS * MICROSTEPS * GEAR_DIV) / 360.0f);
      Serial.print("Преобразование угла поворота: ");
      Serial.print(scan_turn_delta);
      Serial.print(" -> ");
      Serial.println(delta_steps);

      // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
      // stepper.setMaxSpeed(stepSpeed);              
      // stepper.setAcceleration(accelStep);
      //scan_number_of_turns_to_do = 4;

      stepper.setMaxSpeed(MOTOR_MICRO_MAX_SPEED);              
      stepper.setAcceleration(MOTOR_MICRO_DEF_ACCEL);

      Serial.println("Вход в процесс сканирования");
      scanning_action(delta_steps);
      Serial.println("Выход из процесса сканирования");

      scan_start_request = false;
      scan_abort_request = false;
      scan_continue_request = false;

      scan_in_progress = false;
      scan_number_of_turns_completed = 0;

      stepper.setSpeed(0);
      digitalWrite(MOTOR_LED_PIN, LOW);
      digitalWrite(DRV_EN_PIN, HIGH);
    }

    vTaskDelay(pdMS_TO_TICKS(200));
  }

  stepper.setSpeed(0);
  digitalWrite(MOTOR_LED_PIN, LOW);
  digitalWrite(DRV_EN_PIN, HIGH);
}

void scanning_action(int32_t delta_steps) {
  bool prev_scan_outp = !scanning_outp_is_position;
  bool scan_outp_changed = false;

  int curr_degree_angle = 0;
  int curr_position = 0;
  
  scan_continue_request = true;
  scan_in_progress = false;

  DeviceMode current_device_mode_copy;

  while (1) {

    // Смена вывода
    xSemaphoreTake(xMutex, portMAX_DELAY);
    current_device_mode_copy = current_device_mode; 
    scan_outp_changed = (prev_scan_outp != scanning_outp_is_position);
    xSemaphoreGive(xMutex);

    if (current_device_mode_copy != SCAN_MODE) {
      // по хорошему сообщить о кидалове
      break;
    }

    if (scan_outp_changed) {

      Serial.println("Обновление вывода в scanning process");
      prev_scan_outp = scanning_outp_is_position;

      if (prev_scan_outp) {
        drawScanModeScreen(prev_scan_outp, true, curr_position, scan_number_of_turns_to_do);
      } else {
        drawScanModeScreen(prev_scan_outp, true, curr_degree_angle, scan_number_of_turns_to_do * scan_turn_delta);
      }
    } 

    stepper.run();

    // Начало движения 
    if (!scan_in_progress && scan_continue_request) {
        Serial.println("Старт движения");
        digitalWrite(DRV_EN_PIN, LOW);
        stepper.moveTo(stepper.currentPosition() + delta_steps);
        scan_in_progress = true;
        scan_continue_request = false; 
        digitalWrite(MOTOR_LED_PIN, HIGH);
    }

    // Проверка завершения движения
    if (scan_in_progress && !stepper.isRunning()) {
        Serial.println("Стоп машина");
        scan_in_progress = false;
        digitalWrite(MOTOR_LED_PIN, LOW);
        digitalWrite(DRV_EN_PIN, HIGH);

        curr_degree_angle += scan_turn_delta;
        curr_position++;

        if (prev_scan_outp) {
          drawScanModeScreen(prev_scan_outp, true, curr_position, scan_number_of_turns_to_do);
        } else {
          drawScanModeScreen(prev_scan_outp, true, curr_degree_angle, scan_number_of_turns_to_do * scan_turn_delta);
        }
        sendResponse(0x01, (uint16_t) curr_position);

        if (curr_position == scan_number_of_turns_to_do) {
          Serial.println("Сделал все повороты");
          break;
        }
    }

    // Проверка запроса на прерывание движения
    if (scan_abort_request) {
      Serial.println("Поступил запрос на прерывание движения");
      break;
    }

    vTaskDelay(pdMS_TO_TICKS(1));
  }

}

//   if (current_device_mode != SCAN_MODE) return;

//   static bool prevScanInProgress = false;
//   static int prevK = -1, prevN = -1;

//   while (current_device_mode == SCAN_MODE) {
//     if (!scan_start_request) {
//       // Статус: ожидание начала
//       drawScanModeScreen(false, -1, -1, scanning_outp_is_position);
//       vTaskDelay(pdMS_TO_TICKS(100));
//       continue;
//     }

//     scan_in_progress = true;
//     scan_number_of_turns_completed = 0;

//     stepper.setMaxSpeed(motor_scan_speed);
//     stepper.setAcceleration(motor_scan_acceleration);

//     for (int i = 0; i < scan_number_of_turns_to_do; i++) {
//       if (current_device_mode != SCAN_MODE || scan_abort_request) break;

//       // Вращение
//       digitalWrite(MOTOR_LED_PIN, HIGH);
//       stepper.moveTo(stepper.currentPosition() + scan_turn_delta);
//       while (stepper.distanceToGo() != 0) {
//         if (current_device_mode != SCAN_MODE || scan_abort_request) break;
//         stepper.run();
//         drawScanModeScreen(true,
//           scanning_outp_is_position ? (scan_number_of_turns_completed + 1) : stepper.currentPosition(),
//           scanning_outp_is_position ? scan_number_of_turns_to_do : stepper.currentPosition() + stepper.distanceToGo(),
//           scanning_outp_is_position
//         );
//       }
//       digitalWrite(MOTOR_LED_PIN, LOW);

//       if (current_device_mode != SCAN_MODE || scan_abort_request) break;

//       scan_number_of_turns_completed++;

//       // Отправка сообщения клиенту
//       if (clientConnected && activeClient.connected()) {
//         uint8_t response[3];
//         response[0] = 1;
//         uint16_t val = scan_number_of_turns_completed;
//         response[1] = (val >> 8) & 0xFF;
//         response[2] = val & 0xFF;
//         activeClient.write(response, 3);
//       }

//       // Ожидание запроса продолжения
//       while (!scan_continue_request) {
//         if (current_device_mode != SCAN_MODE || scan_abort_request) break;
//         vTaskDelay(pdMS_TO_TICKS(10));
//       }
//       scan_continue_request = false;
//     }

//     // Завершение по прерыванию или окончанию
//     if (scan_abort_request || current_device_mode != SCAN_MODE) {
//       if (clientConnected && activeClient.connected()) {
//         uint8_t response[3] = {2, 0, 0};
//         activeClient.write(response, 3);
//       }
//     }

//     scan_in_progress = false;
//     scan_start_request = false;
//     scan_abort_request = false;
//     scan_continue_request = false;
//     scan_number_of_turns_to_do = 0;
//     scan_number_of_turns_completed = 0;
//     stepper.stop();

//     drawScanModeScreen(false, -1, -1, scanning_outp_is_position);
//   }

//   digitalWrite(MOTOR_LED_PIN, LOW);
//   stepper.stop();

void drawScanModeScreen(bool pos_outp, bool inProgress, int pos, int total) {
  display.clearDisplay();

  // Заголовок
  drawBoldText(0, 0 + VERTICAL_OFFSET, "Scanning mode");

  // Статус сканирования
  display.setCursor(0, 16 + VERTICAL_OFFSET);
  display.print("Scanning: ");
  display.println(inProgress ? "In Progress" : "Disabled");

  // Надпись "Position:"
  display.setCursor(0, 28 + VERTICAL_OFFSET);
  display.println(pos_outp ? "Position:" : "Rotation Angle:");

  // Центрированная строка с текущей позицией
  char buffer[16];
  if (inProgress) {
    snprintf(buffer, sizeof(buffer), "%d/%d", pos, total);
  } else {
    snprintf(buffer, sizeof(buffer), "- / -");
  }

  int textWidth = 6 * strlen(buffer);  // ширина текста
  int xCentered = (SCREEN_WIDTH - textWidth) / 2;
  drawBoldText(xCentered, 40 + VERTICAL_OFFSET, buffer);

  display.display();
}


// Отрисовка экрана с информацией
void info_outp() {

  display.clearDisplay();

  drawBoldText(0, 0 + VERTICAL_OFFSET, "Device Info");

  display.setCursor(0, 16 + VERTICAL_OFFSET);
  display.println("Motor driver: DRV8825");

  display.setCursor(0, 26 + VERTICAL_OFFSET);
  display.println("Motor: Nema 17hs8401");

  display.setCursor(0, 36 + VERTICAL_OFFSET);
  display.println("Display: OLEDv2-0.96");

  display.setCursor(0, 46 + VERTICAL_OFFSET);
  display.println("MCU: Esp32-WROOM-32");

  display.display();

  //while (current_device_mode == DEVICE_INFO) {
  DeviceMode current_device_mode_copy;
  while (true) {

    xSemaphoreTake(xMutex, portMAX_DELAY);
    current_device_mode_copy = current_device_mode;
    xSemaphoreGive(xMutex);

    if (current_device_mode_copy != DEVICE_INFO) {
      break;
    }

    delay(100);
  }
}







// // Тест мотора
// void motor_test() {
//   const int POT_CENTER = POT_MAX_VALUE / 2;
//   //const int CENTER_TOLERANCE = 50;  // мёртвая зона вокруг центра
//   //const float DEG_PER_STEP = 1;//0.6f;
//   //const float DEG_PER_STEP = 0.225f;             // Для 1/8 шага (1.8° / 8)
//   const int SPEED_LEVELS = 36;      // Количество уровней скорости (дискретизация)
//   const int DISPLAY_UPDATE_INTERVAL = 200;  // Обновлять дисплей раз в 100 мс

//   // Отрисовка экрана
//   display.clearDisplay();
//   drawBoldText(0, 0 + VERTICAL_OFFSET, "Motor test");
//   display.setCursor(0, 16 + VERTICAL_OFFSET);
//   display.println("Rotation speed:");
//   display.display();

//   // Настройка мотора
//   digitalWrite(DRV_EN_PIN, LOW);
//   stepper.setMaxSpeed(MOTOR_MICRO_MAX_SPEED);      // Максимальная скорость (шагов в секунду)
//   stepper.setAcceleration(MOTOR_MICRO_DEF_ACCEL);  // Ускорение

//   //digitalWrite(MOTOR_LED_PIN, test_mode_motor_is_running);

//   // Для троттлинга обновления дисплея
//   uint32_t last_display_update = 0;
//   int last_displayed_speed = 0;

//   DeviceMode current_device_mode_copy;
//   bool test_mode_motor_is_running_copy;

//   //while (current_device_mode == MOTOR_TEST) {
//   while (true) {

//     taskENTER_CRITICAL(&mux);
//     current_device_mode_copy = current_device_mode;
//     test_mode_motor_is_running_copy = test_mode_motor_is_running;
//     taskEXIT_CRITICAL(&mux);

//     if (current_device_mode_copy != MOTOR_TEST) {
//       break;
//     }

//     int potValue = analogRead(POTENCIOMETER_PIN);
//     int delta = potValue - POT_CENTER;
//     int stepSpeed = 0;

//     // if (abs(delta) > CENTER_TOLERANCE && test_mode_motor_is_running) {
//     //if (abs(delta) > CENTER_TOLERANCE) {

//       // Преобразуем в диапазон -MAX_SPEED … +MAX_SPEED
//       delta = constrain(delta, -POT_CENTER, POT_CENTER);
      
//       //stepSpeed = map(delta, -POT_CENTER, POT_CENTER, -MOTOR_MICRO_MAX_SPEED, MOTOR_MICRO_MAX_SPEED);
//       //stepSpeed = constrain(stepSpeed, -MOTOR_MICRO_MAX_SPEED, MOTOR_MICRO_MAX_SPEED);
//       int speed_level = map(abs(delta), 0, POT_CENTER, 0, SPEED_LEVELS);
//       stepSpeed = (delta > 0 ? 1 : -1) * map(speed_level, 0, SPEED_LEVELS, 0, MOTOR_MICRO_MAX_SPEED);

//       //digitalWrite(MOTOR_LED_PIN, HIGH);

//     //} else {
//     //  stepSpeed = 0;
//     //  digitalWrite(MOTOR_LED_PIN, LOW);
//     //}

//     stepper.setSpeed(stepSpeed);
//     stepper.runSpeed();

//     // Расчёт скорости в градусах/сек
//     // int motor_test_speed = round(stepSpeed * DEG_PER_STEP);

//     // Пересчёт скорости в градусы/сек
//     float deg_per_sec = (float)stepSpeed * 360.0 / (MOTOR_STEPS * MICROSTEPS * GEAR_DIV);
//     int rounded_deg = round(deg_per_sec);

//     // Обновляем дисплей, только если:
//     // 1) Прошло >100 мс И 2) Скорость изменилась
//     if (millis() - last_display_update >= DISPLAY_UPDATE_INTERVAL && rounded_deg != last_displayed_speed) {
//       char buffer[20];
//       snprintf(buffer, sizeof(buffer), " %d degree/sec", rounded_deg);  //motor_test_speed); degree

//       int textWidth = 6 * strlen(buffer);
//       int xCentered = (SCREEN_WIDTH - textWidth) / 2;

//       //drawBoldText(xCentered, 32 + VERTICAL_OFFSET, buffer);
//       int y = 32 + VERTICAL_OFFSET;

//       display.fillRect(0, y, SCREEN_WIDTH, 10, BLACK);  // Типо очищаем строку
//       drawBoldText(xCentered, y, buffer);
//       display.display();

//       last_display_update = millis();
//       last_displayed_speed = rounded_deg;
//     }

//     vTaskDelay(1);
//   }

//   stepper.setSpeed(0);
//   digitalWrite(MOTOR_LED_PIN, LOW);
//   digitalWrite(DRV_EN_PIN, HIGH);
// }
