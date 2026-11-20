# A Python list repeats when you multiply it. A NumPy array does the maths.
import numpy as np

print([4, 5, 6] * 2)                 # a list: repeated

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])

print("Votes (Array):", votes)
print(type(votes))                   # <class 'numpy.ndarray'>: n-dimensional array

print(votes * 2)                     # an array: every vote doubled
print(votes.sum(), votes.max(), votes.mean())
