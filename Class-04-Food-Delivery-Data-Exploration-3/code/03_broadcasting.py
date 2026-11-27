# Broadcasting: a smaller array stretches to fit, when the shapes fit from the right
# (equal, or one of them is 1). Nothing is copied.
import numpy as np

# Start in 1D: one number, same length, a length that does not fit
prices = np.array([100, 200, 300])
print(prices + 20)                        # the 20 stretches to 3
print(prices + np.array([10, 20, 30]))    # same length: no stretch
print(prices + np.array([20]))            # length 1 stretches
# prices + np.array([10, 20])   ValueError: shapes (3,) (2,)

carts = np.array([[2, 1, 3], [1, 4, 0], [0, 2, 2]])
price_lists = np.array([[250, 220], [40, 50], [60, 70]])
bills = carts @ price_lists

# Pricing: Rs 30 delivery fee on every bill
print(bills + 30)

# Packaging per dish, added to the menus of two outlets: (2, 3) + (3,)
menu = np.array([[250, 40, 60],
                 [230, 45, 65]])
packing = np.array([10, 5, 15])
print(menu + packing)

# A sale preview: dishes at 100, 200, 400, each at 10, 20, 50% off. (3, 1) * (1, 3) gives (3, 3)
prices = np.array([[100], [200], [400]])
off = np.array([[0.9, 0.8, 0.5]])
print(prices.shape, off.shape)
print(prices * off)

# Shapes that fit, and one that does not
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a + np.array([[10], [20]]))      # (2, 3) and (2, 1): works
# a + np.array([[10, 20]])             # (2, 3) and (1, 2): ValueError

A = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
print(A + b)                           # (3, 1) and (3,): a 3 x 3 table
