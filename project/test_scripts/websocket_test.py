import websocket

def on_message(ws, message):
    print(f"Ответ от ESP: {message}")

def on_error(ws, error):
    print(f"Ошибка: {error}")

def on_close(ws, close_status_code, close_msg):
    print("Соединение закрыто")

def on_open(ws):
    while True:
        cmd = input("Введ" \
        "ите команду (on/off/exit): ").strip()
        if cmd == "exit":
            ws.close()
            break
        ws.send(cmd)

esp_ip = "ws://192.168.4.1/ws"

ws = websocket.WebSocketApp(
    esp_ip,
    on_open=on_open,
    on_message=on_message,
    on_error=on_error,
    on_close=on_close
)

websocket.enableTrace(False)
ws.run_forever()
