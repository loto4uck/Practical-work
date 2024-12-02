numbers = input("Enter 5 digits separated by spaces: ").split()
numbers = [int(num) for num in numbers]
reversed_numbers = numbers[::-1]
result = ''.join(map(str, reversed_numbers))
print("Resulting number:", result)
