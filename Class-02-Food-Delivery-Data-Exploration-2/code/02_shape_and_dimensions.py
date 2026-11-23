# shape, ndim and size describe how an array is laid out.
import numpy as np

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])

# Note: intentionally messy data. Each cost is text, with extra quotes inside.
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])

print("Votes array shape:", votes.shape)
print("Votes array dimensions:", votes.ndim)
print("Votes array size:", votes.size)

print("Costs array shape:", costs.shape)
print("Costs array dimensions:", costs.ndim)
print("Costs array size:", costs.size)

# shape and size are not the same. An array with 2 rows and 3 columns:
m = np.array([[1, 2, 3],
              [4, 5, 6]])

print(m.shape)   # (2, 3): the layout, rows then columns
print(m.size)    # 6: every element, counted (2 x 3)
print(m.ndim)    # 2: how many numbers the shape has

# Transpose (.T): every row becomes a column. No value changes.
print(m.T)
print(m.T.shape)

# Take first 5 elements of votes and costs
subset_votes = votes[:5]
subset_costs = costs[:5]

# Create a 2D array: 5 rows, 2 columns (each row: [vote_count, cost])
two_d_data = np.array([
    subset_votes,
    subset_costs
]).T  # transpose so that each row corresponds to a single restaurant

print("2D Array:\n", two_d_data)
print("Shape:", two_d_data.shape)
print("Dimensions:", two_d_data.ndim)
print("Size:", two_d_data.size)

# Read the shape: rows first, then columns.
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print(a.shape, a.size, a.ndim)   # (3, 4) 12 2

# Three levels of brackets: read from the outside in.
# 2 blocks, 3 rows in each block, 2 numbers in each row.
b = np.array([[[1, 2],
               [3, 4],
               [5, 6]],
              [[7, 8],
               [9, 10],
               [11, 12]]])
print(b.shape, b.size, b.ndim)   # (2, 3, 2) 12 3

# Rows of different lengths are not allowed: a 2D array is a full rectangle.
# np.array([[1, 2, 3], [4, 5]])   ->  ValueError: ... inhomogeneous shape ...
# Forced with dtype=object, each row is a plain Python list stored inside the array.
jagged = np.array([[1, 2, 3], [4, 5]], dtype=object)
print(jagged.shape)    # (2,): a row of 2 lists, not a table
print(jagged * 2)      # the lists are repeated, not doubled: no fast maths
