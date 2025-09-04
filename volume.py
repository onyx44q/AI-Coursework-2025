import math
try:
  
    radius_str = input("Enter the radius of the sphere: ")

    # Convert the user's input from a string to a floating-point number
    radius = float(radius_str)

    # Check if the radius is a non-negative number
    if radius < 0:
        print("\nEnter a positive number.")
    else:
        # Calculate the volume of the sphere using the formula V = (4/3) * pi * r^3
        # The exponential operator (**) is used to calculate the cube of the radius.
        volume = (4/3) * math.pi * (radius ** 3)

        print(f"\nThe volume of a sphere with radius  is {volume}.")

except ValueError:
    # If the user's input cannot be converted to a number,
    # this message is printed.
    print("\nEnter a valid number for the radius.")