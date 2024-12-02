# Словник з мовами програмування та їх розробниками
languages = {
    'Python': 'Guido van Rossum',
    'C++': 'Bjarne Stroustrup',
    'Java': 'James Gosling',
    'JavaScript': 'Brendan Eich'
}

# Виведення повідомлень для кожної пари
for language, developer in languages.items():
    print(f"My favorite programming language is {language}. It was created by {developer}.")

# Видалення пари з словника
del languages['Java']

# Виведення словника після видалення
print("\nUpdated dictionary:")
print(languages)
