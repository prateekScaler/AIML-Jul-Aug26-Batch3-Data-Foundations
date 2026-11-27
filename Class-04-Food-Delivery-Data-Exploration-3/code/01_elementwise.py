# Element-wise maths: * + - / pair up the same positions of two arrays.
# An array and one number: the number is used for every element.
import numpy as np

# Our 10 practice restaurants: votes, rating out of 5, cost for two (Rs)
votes = np.array([775, 787, 918, 88, 166,
                  286, 2556, 324, 504, 402])
ratings = np.array([4.1, 4.1, 3.8, 3.7, 3.8,
                    3.8, 4.6, 4.0, 4.2, 4.1])
costs = np.array([800., 800., 800., 300., 600.,
                  600., 600., 700., 550., 500.])

# The manager: a 5.0 from 2 people or a 4.8 from 1000? A trust score uses both.
trust = votes * ratings
print(trust)
# Does the rating change the order? No: the votes (hundreds) swamp the ratings (about 4).
print(np.argsort(trust)[::-1])
print(np.argsort(votes)[::-1])

# Restaurants: every price goes up 5%
new_costs = costs * 1.05
print(new_costs)

# Shapes that do not fit give an error:
# votes * np.array([1, 2, 3, 4])
# ValueError: operands could not be broadcast together with shapes (10,) (4,)

a = np.array([2, 4, 6])
b = np.array([10, 20, 30])
print(a * b + 1)
