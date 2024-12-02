import math

# Координати Пекіна і Києва
x1, y1 = 39.9075000, 116.3972300  # Пекін
x2, y2 = 50.4546600, 30.5238000  # Київ

# Переводимо координати з градусів в радіани
x1, y1 = math.radians(x1), math.radians(y1)
x2, y2 = math.radians(x2), math.radians(y2)

# Обчислюємо відстань між точками
distance = 6371.032 * math.acos(math.sin(x1) * math.sin(x2) + math.cos(x1) * math.cos(x2) * math.cos(y1 - y2))

# Виводимо результат
print(f"Distance between Beijing and Kyiv: {distance:>10.3f} km")
