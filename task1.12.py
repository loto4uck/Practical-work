# Програма для перетворення температури

# Лоточук Максим Сергійович 15.11.2024
# Це програма, яка перетворює температуру з Цельсія в інші шкали
# Вводимо температуру в Цельсії
celsius = float(input("Enter temperature in Celsius: "))

# Перетворюємо в Фаренгейт і Кельвін
fahrenheit = 32 + (9 / 5) * celsius
kelvin = celsius + 273.15

# Виводимо результат
print(f"{celsius:^15.2f}°C = {fahrenheit:^15.2f}°F = {kelvin:^15.2f}K")
