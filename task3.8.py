# Створюємо словник з містами
cities = {
    'Kyiv': {'country': 'Ukraine', 'population': 2800000, 'fact': 'The capital of Ukraine.'},
    'New York': {'country': 'USA', 'population': 8500000, 'fact': 'The city that never sleeps.'},
    'Tokyo': {'country': 'Japan', 'population': 14000000, 'fact': 'Known for its technology and culture.'}
}

# Виведення інформації про міста
for city, info in cities.items():
    print(f"{city}:")
    for key, value in info.items():
        print(f"\t{key}: {value}")
