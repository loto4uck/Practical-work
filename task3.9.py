# Створюємо словник з командами NBA
teams = {
    'NEW YORK KNICKS': [22, 7, 6, 9, 45],
    'LOS ANGELES LAKERS': [23, 15, 4, 4, 50],
    'BOSTON CELTICS': [24, 18, 3, 3, 55]
}

# Виведення статистики команд
for team, stats in teams.items():
    print(f"{team} {stats[0]} {stats[1]} {stats[2]} {stats[3]} {stats[4]}")
