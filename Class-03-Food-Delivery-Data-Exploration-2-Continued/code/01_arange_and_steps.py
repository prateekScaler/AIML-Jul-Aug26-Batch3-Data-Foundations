# np.arange(start, end, step) makes numbers in a row. The end is left out.
# A slice can take a step too: array[start : end : step]. A negative step walks backwards.
import numpy as np

# The app team: IDs for the first 1000 customers in a new city
ids = np.arange(1, 1001)
print(ids[:5])
print(ids[-3:])
print(ids.size)

print(np.arange(1, 1000)[-1])      # 999: the end is left out

# The third number is the step
print(np.arange(100, 1000, 100))   # the product team: cost-for-two filter buttons, 100 to 900
print(np.arange(3, 5, 0.5))        # rating filter 3.0 to 4.5: a decimal step works; Python's range() does not allow it

# Step slicing
votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
a = np.arange(10)    # positions of our 10 restaurants
print(a)
print(a[::2])        # every second element: the quality team visits these
print(votes[::2])    # the votes of those five restaurants
print(a[::-1])       # reversed: newest first, as on the "Your orders" screen
print(a[::-2])       # from the end, every second element
print(a[::-3])       # [9 6 3 0]
print(a[1:8:3])      # [1 4 7]: start at 1, jump by 3, stop before 8
