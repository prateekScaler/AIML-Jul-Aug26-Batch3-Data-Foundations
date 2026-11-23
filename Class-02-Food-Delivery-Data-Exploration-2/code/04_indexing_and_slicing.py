# Indexing picks one element. Slicing picks a part. In 2D: row first, then column.
import numpy as np

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])
two_d_data = np.array([votes[:5], costs[:5]]).T

print(votes[0])
print(votes[-1])
print(votes[-2])

print(votes[[2, 3, 4, 1, 2, 2]])     # a list of positions

a = np.array([100, 200, 300, 400, 500])
print(a[1:4])                        # starts at 1, stops before 4

print("First 5 votes:", votes[:5])
print("First 10 costs:", costs[:10])

# 2D array slicing
print("First 3 rows of the 2D array:\n", two_d_data[:3, :])
print("All votes:", two_d_data[:, 0])
print("All costs:", two_d_data[:, 1])
print("One cell:", two_d_data[1, 0])

print("Last vote:", votes[-1])
print("Second last vote:", votes[-2])

print(two_d_data[:, 0].shape)        # one column of a (5, 2) array

# One column: a number drops a direction, a slice keeps it.
arr = np.array([[1, 2],
                [3, 4],
                [5, 6]])
print(arr[:, 0])          # [1 3 5]: one bracket, a 1D array
print(arr[:, 0].shape)    # (3,)
print(arr[:, 0:1])        # [[1] [3] [5]]: two brackets, a 2D array
print(arr[:, 0:1].shape)  # (3, 1)

# A slice that goes past the end does not fail: it stops at the end.
print(votes[8:20])        # [504 402]
