import socket

ESP_IP = '192.168.206.141'  # IP адрес ESP (точка доступа)
ESP_PORT = 1234         # Порт сервера на ESP

def recv_until_newline(sock, timeout=2.0):
    sock.settimeout(timeout)
    data = b""
    try:
        while not data.endswith(b"\n"):
            part = sock.recv(1024)
            if not part:
                break
            data += part
    except socket.timeout:
        pass
    return data.decode().strip()

def main():
    try:
        # Создаём и подключаем сокет один раз
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5.0)
        s.connect((ESP_IP, ESP_PORT))

        print("Подключено к ESP32. Введите команды (on/off/exit).")

        response = s.recv(1024).decode().strip()
        print(f"Ответ от ESP: {response}")

        while True:
            cmd = input("> ").strip().lower()
            
            if cmd == "exit":
                break

            s.sendall((cmd + "\n").encode())
            try:
                response = s.recv(1024).decode().strip()
                if response:
                    print(f"Ответ от ESP: {response}")
                else:
                    print("(Нет ответа от ESP)")
            except socket.timeout:
                print("Таймаут: ESP не ответил")

    except ConnectionRefusedError:
        print("Ошибка: ESP32 не принимает подключения. Проверьте IP и порт.")
    except Exception as e:
        print(f"Ошибка: {e}")
    finally:
        s.close()  

if __name__ == "__main__":
    main()
