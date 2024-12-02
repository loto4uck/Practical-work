# Створюємо глосарій, де ключі - терміни, а значення - їх визначення
glossary = {
    'Python': 'A high-level programming language.',
    'Algorithm': 'A step-by-step procedure for solving a problem.',
    'Function': 'A block of code that only runs when it is called.',
    'Variable': 'A container for storing data values.'
}

# Виведення глосарію
for word, definition in glossary.items():
    print(f"{word}:\n\t{definition}\n")
