def calculate_average():
    values = [] # Initialize an empty list to store the values
    print("\nPlease enter 5 numeric values:")

    # Loop to get 5 values from the user
    for i in range(5):
        while True: 
            try:
                
                user_input = input(f"Enter value {i + 1}: ")
                # Convert the input to a float (to handle decimals)
                value = float(user_input)
                values.append(value) # Add the valid value to the list
                break # Exit the inner while loop once a valid number is entered
            except ValueError:
               
                print("Invalid input")

    # Calculate the sum of the values in the list
    total_sum = sum(values)
    average = total_sum / len(values) 

    # Display the calculated average
    print(f"\nValues entered: {values}")
    print(f"The average of these values is: {average}") 


while True:
    calculate_average() # Call the function to perform the input and calculation

    while True:
        choice = input("\nDo you want to calculate another average? (yes/no): ").lower()
        if choice in ['yes', 'no']:
            break #
        else:
            print("Invalid choice")

    if choice == 'no':
        print("Exiting")
        break 
    