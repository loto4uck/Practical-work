# Словник з речами в грі
things = {
    'key': 3, 'mace': 1, 'gold coin': 24, 'lantern': 1, 'stone': 10
}

# Виведення інформації про речі
print("Equipment:")
total_things = 0
for item, quantity in things.items():
    print(f"{quantity} {item}")
    total_things += quantity

print(f"Total number of things: {total_things}")
