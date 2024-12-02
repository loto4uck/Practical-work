# Створюємо багаторівневий словник
subjects = {
    'science': {
        'physics': ['nuclear physics', 'optics', 'thermodynamics'],
        'computer science': {},
        'biology': {}
    },
    'humanities': {},
    'public': {}
}

# Виведення значень
print("Subjects in science:", subjects['science'])
print("Physics topics:", subjects['science']['physics'])
