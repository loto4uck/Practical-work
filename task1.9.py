# Програма для знаходження суми цифр числа

# Вводимо число
number = 6259

# Розкладаємо число на цифри
sum_digits = sum(int(digit) for digit in str(number))

# Виводимо результат
print(f"{' + '.join(str(number))} = {sum_digits}")
