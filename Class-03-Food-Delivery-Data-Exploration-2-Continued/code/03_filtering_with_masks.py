# A mask is an array of True and False. Inside the brackets it filters:
# only the elements where the mask is True come through.
import numpy as np

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])
clean_costs = np.char.strip(costs, "'").astype(float)

# The product team: a "Popular" filter, 500 votes or more
print(votes >= 500)              # the mask: one True or False for each restaurant
print(votes[votes >= 500])       # the filter: only the votes that pass
print((votes >= 500).sum())      # True counts as 1, so this counts how many passed: 5

# The manager: are popular places more expensive?
# A mask from one array can filter another, when the two arrays line up
print(clean_costs[votes >= 500])
print(clean_costs[votes >= 500].mean())   # popular places: 710.0
print(clean_costs[votes < 500].mean())    # the rest: 540.0

# Marketing: a "Popular and affordable" collection
# Two conditions: each in round brackets, joined by &
mask = (votes > 500) & (clean_costs < 700)
print(votes[mask])
print(clean_costs[mask])

# Your turn (another business, the same tool): a fitness band records the mood and the step count of each user
mood = np.array(["Happy", "Sad", "Happy", "Sad",
                 "Happy", "Happy", "Sad", "Happy"])
steps = np.array([9200, 4100, 10500, 5200,
                  8700, 11000, 3900, 9600])
print(steps[mood == "Happy"].mean())
print(steps[mood == "Sad"].mean())
