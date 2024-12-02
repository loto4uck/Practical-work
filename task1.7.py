# Програма для обчислення тривалості події в годинах, хвилинах і секундах

# Припустимо, канікули тривали 10 днів
days = 10

# Обчислюємо тривалість у годинах, хвилинах, секундах
hours = days * 24
minutes = hours * 60
seconds = minutes * 60

# Виводимо результат
print(f"{days} days is:")
print(f"{hours:<10} hours")
print(f"{minutes:<10} minutes")
print(f"{seconds:<10} seconds")
