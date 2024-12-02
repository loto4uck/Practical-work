# Створюємо англо-німецький словник
e2g = {
    'stork': 'storch',
    'hawk': 'falke',
    'woodpecker': 'specht',
    'owl': 'eule'
}

# Виведення німецької версії слова "owl"
print(f"The German word for 'owl' is {e2g['owl']}.")

# Додавання нових слів
e2g['cat'] = 'katze'
e2g['dog'] = 'hund'

# Виведення словника та його ключів і значень
print("\nUpdated dictionary:")
print(e2g)

print("\nKeys:", list(e2g.keys()))
print("Values:", list(e2g.values()))
