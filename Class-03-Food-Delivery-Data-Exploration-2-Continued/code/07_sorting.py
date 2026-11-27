# np.sort gives a sorted copy. Reverse it with [::-1] for the largest first.
# np.argsort gives the positions in sorted order: use them on a parallel array.
# In 2D, axis=1 sorts each row and axis=0 sorts each column, on its own.
import numpy as np

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])
clean_costs = np.char.strip(costs, "'").astype(float)
data = np.column_stack((votes, clean_costs))

print(np.sort(votes))
print(np.sort(votes)[::-1])
print(np.argsort(votes))

# The manager: the most popular places, and what they cost
order = np.argsort(votes)[::-1]     # most votes first
print(votes[order][:3])
print(clean_costs[order][:3])       # the costs stay with their restaurants

m = np.array([[34, 11,  8],
              [ 7, 45, 18],
              [ 9, 23, 20]])
print(np.sort(m, axis=1))
print(np.sort(m, axis=0))

print(np.sort(data, axis=0)[:3])            # each column sorted apart: rows broken
print(data[np.argsort(data[:, 0])][:3])     # whole rows, ordered by votes

# Numbers stored as text sort like words: a "low to high" sort puts 1200 first
prices = np.array(["800", "1200", "300", "550"])
print(np.sort(prices))
