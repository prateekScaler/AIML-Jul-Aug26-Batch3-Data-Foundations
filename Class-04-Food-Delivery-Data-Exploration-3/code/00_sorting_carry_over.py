# Carry-over from Class 3: sorting.
# np.sort gives the values (smallest first; [::-1] for largest first).
# np.argsort gives the positions: use them on parallel arrays to keep each restaurant together.
import numpy as np

votes = np.array([775, 787, 918, 88, 166,
                  286, 2556, 324, 504, 402])
costs = np.array([800., 800., 800., 300., 600.,
                  600., 600., 700., 550., 500.])

# The product team: "Sort by popularity", most votes first
print(np.sort(votes))
print(np.sort(votes)[::-1])
# np.sort(votes, reverse=True)
# TypeError: sort() got an unexpected keyword argument 'reverse'

# The manager: the three most popular places, and what they cost
print(np.argsort(votes))
order = np.argsort(votes)[::-1]
print(votes[order][:3])
print(costs[order][:3])

# The manager's sheet, ranked by votes: sort whole rows, not each column
data = np.column_stack((votes, costs))
print(np.sort(data, axis=0)[:3])            # each column sorted apart: rows broken
col = data[:, 0]          # 1. the votes column
order = np.argsort(col)   # 2. row numbers, fewest votes first
ranked = data[order]      # 3. whole rows, in that order
print(order)
print(ranked[:3])

# Numbers stored as text sort like words
prices = np.array(["800", "1200", "300", "550"])
print(np.sort(prices))
