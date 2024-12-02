# Програма для перетворення температури з Цельсія в Фаренгейт і Кельвін

# Вводимо температуру в Цельсії
celsius = 25

# Перетворюємо в Фаренгейт і Кельвін
fahrenheit = 32 + (9 / 5) * celsius
kelvin = celsius + 273.15

# Виводимо результат
print(f"{celsius:^15.2f}°C = {fahrenheit:^15.2f}°F = {kelvin:^15.2f}K")
