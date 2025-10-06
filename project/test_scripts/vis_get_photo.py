import numpy as np
import cv2

# === Загрузка файлов ===
depth_background = np.load('D:\\diploma\\rgb_n_depth\\teapot_depth_background.npy')
depth_object = np.load('D:\\diploma\\rgb_n_depth\\teapot_depth_0.npy')
color_image = cv2.imread('D:\\diploma\\rgb_n_depth\\teapot_color_0.png')

# === Преобразование цветного изображения ===
color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

# === Порог фильтрации ===
max_distance_mm = 1000  # Дальность в мм

# === Визуализация глубины с объектом ===
depth_normalized = cv2.normalize(depth_object, None, 0, 255, cv2.NORM_MINMAX)
depth_colored = cv2.applyColorMap(depth_normalized.astype(np.uint8), cv2.COLORMAP_JET)

# === Визуализация фона ===
depth_bg_normalized = cv2.normalize(depth_background, None, 0, 255, cv2.NORM_MINMAX)
depth_bg_colored = cv2.applyColorMap(depth_bg_normalized.astype(np.uint8), cv2.COLORMAP_JET)

# === Фильтрация по дальности ===
mask_far_obj = depth_object > max_distance_mm
mask_far_bg = depth_background > max_distance_mm
depth_colored[mask_far_obj] = [255, 255, 255]
depth_bg_colored[mask_far_bg] = [255, 255, 255]

# === Определение значения верхнего левого пикселя ===
top_left_value = depth_object[0, 0]
print(f"[INFO] Значение верхнего левого пикселя: {top_left_value}")

# === Маска совпадений с этим значением ===
mask_top_left = (depth_object == top_left_value)
highlighted_depth_colored = depth_colored.copy()
highlighted_depth_colored[mask_top_left] = [0, 0, 255]  # Чисто красный

# === ВЫЧИТАНИЕ ФОНА ===
depth_diff = cv2.absdiff(depth_object, depth_background)

# === БИНАРИЗАЦИЯ ===
_, binary_mask = cv2.threshold(depth_diff, 30, 255, cv2.THRESH_BINARY)  # 30 мм — чувствительность
binary_mask = binary_mask.astype(np.uint8)

# === ДОБАВЛЕННАЯ ФИЛЬТРАЦИЯ по дальности ===
binary_mask[depth_object > max_distance_mm] = 0  # Далёкие пиксели → чёрные

binary_mask[depth_object == top_left_value] = 0

# === ОБРАБОТКА МАСКИ: Размытие + Морфология ===

# # 1. Размытие (сглаживает резкие переходы, помогает морфологии)
# blurred_mask = cv2.GaussianBlur(binary_mask, (5, 5), 0)

# # 2. Бинаризация снова после размытия (чтобы вернуть чёткие границы)
# _, blurred_mask = cv2.threshold(blurred_mask, 127, 255, cv2.THRESH_BINARY)

blurred_mask = binary_mask.copy()

# 3. Морфологические операции
kernel = np.ones((3, 3), np.uint8)

cleaned_mask = blurred_mask.copy()

cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=2)

cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=11)

cleaned_mask = cv2.erode(cleaned_mask, kernel, iterations=3)

# === Поиск контуров на обработанной маске ===
contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Проверка на наличие контуров
if contours:
    # Найдём самый большой контур по площади
    largest_contour = max(contours, key=cv2.contourArea)

    # Копия изображения, чтобы не рисовать прямо на исходном
    depth_with_contour = highlighted_depth_colored.copy()

    # Нарисуем самый большой контур — красным цветом, толщина 2
    cv2.drawContours(depth_with_contour, [largest_contour], -1, (0, 255, 0), 2)
else:
    print("[INFO] Контуры не найдены")


# === Отображение ===
cv2.imshow("Color Image", color_bgr)
cv2.imshow("Depth with Object (Filtered + Highlight)", highlighted_depth_colored)
cv2.imshow("Background Depth (Filtered)", depth_bg_colored)
cv2.imshow("Binary Foreground Mask", binary_mask)

# === Показываем маску после обработки ===
cv2.imshow("Processed Mask", cleaned_mask)

# Покажем результат
cv2.imshow("Depth with Largest Contour", depth_with_contour)

cv2.waitKey(0)
cv2.destroyAllWindows()
