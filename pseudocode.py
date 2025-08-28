
x = 0  
y = 20 


# The loop continues as long as y is NOT less than 6 (i.e., y is greater than or equal to 6)
while y >= 6:
    y -= 4  # SUBTRACT 4 FROM y
    # Ensure y is not zero before division to prevent ZeroDivisionError
    if y != 0:
        x += (2 / y) # ADD 2/y TO x
    else:
        # y goes 20 -> 16 -> 12 -> 8 -> 4, so it won't be zero until after the loop condition changes.
     
        print("Loop might not behave as intended if y reaches 0.")
        break 

# UNTIL y IS LESS THAN 6 (loop terminates when this condition becomes true)

print(x)
