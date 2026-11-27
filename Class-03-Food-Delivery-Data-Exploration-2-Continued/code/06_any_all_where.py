# np.any: is at least one True? np.all: is every one True? Each gives one True or False.
# np.where(condition, a, b) is an if-else for every element.
# np.where(condition) gives the positions where the condition is True.
import numpy as np

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])
clean_costs = np.char.strip(costs, "'").astype(float)

# The manager: check the data before the weekly slide
print(np.any(clean_costs > 3000))
print(np.all(clean_costs < 5000))

ratings = np.array([4.5, 4.3, 4.1])
print(np.all(ratings > 4.2))        # a "Top rated" badge: 4.1 fails, so False

ratings = np.array([4.9, 4.1, 4.8])     # delivery partners' ratings
print(np.where(ratings >= 4.2, "Green Flag", "Red Flag"))

tags = np.where(votes >= 500, "Popular", "New")    # a badge for every card
print(tags)

print(np.where(votes >= 500))           # the positions
print(np.where(clean_costs > 1000))     # no match: an empty array
