ch = input("Enter a character: ")

if len(ch) != 1:
    print("Please enter a single character.")
elif ch.islower():
    print(f"'{ch}' is lowercase.")
elif ch.isupper():
    print(f"'{ch}' is uppercase.")
else:
    print(f"'{ch}' is not an alphabetic character.")