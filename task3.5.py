# Створюємо словники для домашніх тварин
pets = [
    {'name': 'Alex', 'pet': 'dog', 'owner': 'Alex'},
    {'name': 'Bella', 'pet': 'cat', 'owner': 'Lucy'},
    {'name': 'Charlie', 'pet': 'hamster', 'owner': 'Tom'}
]

# Виведення інформації про кожного домашнього улюбленця
for pet in pets:
    print(f"{pet['owner']} is the owner of a pet - a {pet['pet']}.")
