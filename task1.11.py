# Програма для введення відстані та конвертації одиниць вимірювання

# Вводимо відстань в метрах
distance_in_meters = float(input("Enter distance in meters: "))

# Переводимо в інші одиниці
inches = distance_in_meters * 39.3701
feet = distance_in_meters * 3.28084
miles = distance_in_meters * 0.000621371

# Виводимо результат
print(f"{distance_in_meters} meters is:")
print(f"{inches:.2f} inches")
print(f"{feet:.2f} feet")
print(f"{miles:.2f} miles")
