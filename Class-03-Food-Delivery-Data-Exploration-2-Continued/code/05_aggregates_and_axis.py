# Aggregates turn many numbers into one: sum, mean, max, min, std.
# On a table, axis=0 gives one answer per column (the rows disappear),
# axis=1 gives one answer per row (the columns disappear).
import numpy as np

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])
clean_costs = np.char.strip(costs, "'").astype(float)
data = np.column_stack((votes, clean_costs))

# The manager's weekly slide
print(data[:, 0].sum())     # total votes
print(data[:, 1].mean())    # average cost for two
print(data[:, 0].max())     # most votes
print(data[:, 1].min())     # cheapest

# Operations: can a restaurant promise "30 minutes"? Same average, different spread
a = np.array([28, 30, 32])      # restaurant A, last three deliveries, in minutes
b = np.array([10, 30, 50])      # restaurant B
print(a.mean(), b.mean())
print(a.std().round(2), b.std().round(2))

# Operations: orders of 3 restaurants (rows) on Fri, Sat, Sun (columns)
orders = np.array([[20, 35, 30],
                   [12, 18, 15],
                   [25, 40, 32]])
print(orders.sum())               # 227: everything
print(np.sum(orders, axis=0))     # one total per day: plan riders for the busiest
print(np.sum(orders, axis=1))     # one total per restaurant

print(data.mean(axis=0))    # average votes and average cost, in one line
print(data.max(axis=0))     # the top of each column: not one restaurant

# Your turn (a restaurant owner): the average votes of the restaurants that cost 800 for two
print(votes[clean_costs == 800].mean())
