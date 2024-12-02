# Створюємо словник з річками та регіонами
rivers = {
    'Amazon': 'South America',
    'Nile': 'Africa',
    'Mississippi': 'North America'
}

# Виведення повідомлень про річки
for river, region in rivers.items():
    print(f"The {river} runs through {region}.")
