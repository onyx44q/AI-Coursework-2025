
seconds_in_minute = 60
minutes_in_hour = 60
hours_in_day = 24

# Calculate the total number of seconds in a day
seconds_in_a_day = seconds_in_minute * minutes_in_hour * hours_in_day

try:
   
    days_str = input("Enter the number of days: ")

    # Convert the user's input from a string to a floating-point number
    # This allows for fractional days (e.g., 1.5 days)
    days_float = float(days_str)

    # Calculate the total number of seconds
    total_seconds = days_float * seconds_in_a_day

    print(f"\nThere are {total_seconds:} seconds in {days_float} day(s).")

except ValueError:
    # If the user enters text that cannot be converted to a number,
    print("\nError: Invalid input")
