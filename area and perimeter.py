def calculate_area(side_length):
    # The formula for the area of a square is side * side or side^2
    return side_length ** 2

def calculate_perimeter(side_length):
    # The formula for the perimeter of a square is 4 * side
    return 4 * side_length

try:
    side_str = input("Enter the side length of the square: ")

    # Convert the user's input from a string to a floating-point number.
    side = float(side_str)

    # Check if the side length is a positive number.
    # Area and perimeter calculations are not meaningful for negative lengths.
    if side <= 0:
        print("\nThe side length must be a positive number.")
    else:
        # Call the functions to compute the area and perimeter.
        area = calculate_area(side)
        perimeter = calculate_perimeter(side)

        print(f"The area is: {area}")
        print(f"The perimeter is: {perimeter}")

except ValueError:
    print("\nEnter a valid number for the side length.")





