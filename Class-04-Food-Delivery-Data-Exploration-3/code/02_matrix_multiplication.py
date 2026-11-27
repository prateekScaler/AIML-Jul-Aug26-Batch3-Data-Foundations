# Matrix multiplication: a row times a column, multiplied in pairs, then added.
# (A, B) @ (B, C) gives (A, C): the middle numbers must match.
import numpy as np

# The app: one cart (2 biryani, 1 naan, 3 lassi) and the prices
prices = np.array([250, 40, 60])
cart = np.array([2, 1, 3])
print(cart * prices)     # each line of the bill
print(cart @ prices)     # the bill: 500 + 40 + 180 = 720

# Three customers checking out: one row per cart
carts = np.array([[2, 1, 3],
                  [1, 4, 0],
                  [0, 2, 2]])
print(carts @ prices)    # one bill per customer

# The same carts at two restaurants: one column per restaurant
price_lists = np.array([[250, 220],
                        [ 40,  50],
                        [ 60,  70]])
bills = carts @ price_lists
print(bills)             # row = customer, column = restaurant

# A price list for 4 dishes does not fit 3-dish carts:
# carts @ np.array([250, 40, 60, 30])
# ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 ...

# @, np.dot and np.matmul agree on matrices
print(np.array_equal(bills, np.dot(carts, price_lists)),
      np.array_equal(bills, np.matmul(carts, price_lists)))

# Only np.dot accepts a single number (a @ 5 and np.matmul(a, 5) give an error)
a = np.array([1, 2, 3, 4])
print(np.dot(a, 5))

# * against @
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(A * B)
print(A @ B)

# The ranking team: 70% rating, 30% popularity (votes in thousands)
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])
ratings = np.array([4.1, 4.1, 3.8, 3.7, 3.8, 3.8, 4.6, 4.0, 4.2, 4.1])
features = np.array([ratings, votes / 1000]).T
weights = np.array([0.7, 0.3])
score = features @ weights
print(features[:3])
print(score.round(2))

# Why votes / 1000? Without it, the votes swamp the rating: the plain vote order again.
raw = np.array([ratings, votes]).T @ weights
print(np.argsort(raw)[::-1])
print(np.argsort(score)[::-1])
