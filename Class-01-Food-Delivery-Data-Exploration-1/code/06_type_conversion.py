# astype() changes the type of every element in an array.
import numpy as np

print("500" + "4")       # text: joined, not added
print(500 + 4)           # numbers: added

a = np.array(['1.2', '2.5', '3.6', '4.8'])
print("Type of a: ", type(a))
print("Type of elements of a: ", a.dtype)

a = a.astype(float)
print(a.dtype)
print(a)

arr = np.array([1.7, 2.8, 3.9])
print(arr.astype(int))   # the decimal part is cut off, not rounded

# Only text that looks like a number can be converted
try:
    np.array(["abc", "xyz"]).astype(float)
except ValueError as e:
    print("ValueError:", e)

# Our messy costs have extra quotes inside, so they fail too
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])
try:
    costs.astype(float)
except ValueError as e:
    print("ValueError:", e)

# Clean first, then convert
clean_costs = np.char.strip(costs, "'").astype(float)
print(clean_costs)
print("Average cost for two:", clean_costs.mean())
