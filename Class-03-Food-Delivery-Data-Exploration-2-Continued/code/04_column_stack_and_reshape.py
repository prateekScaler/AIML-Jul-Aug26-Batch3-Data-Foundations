# np.column_stack puts arrays side by side as columns: one row per restaurant.
# reshape refills the same numbers into new rows and columns. rows x columns must equal the size.
# -1 lets NumPy work out one side.
import numpy as np

votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
costs = np.array(["'800.0'" ,"'800.0'", "'800.0'", "'300.0'", "'600.0'", "'600.0'", "'600.0'", "'700.0'" ,"'550.0'", "'500.0'"])
clean_costs = np.char.strip(costs, "'").astype(float)

# The manager: one sheet, one row per restaurant
data = np.column_stack((votes, clean_costs))
print(data.shape)       # (10, 2): 10 restaurants, 2 columns
print(data[:3])         # the votes became floats: one type only

# The app team: restaurant cards in a grid on the home screen
ids = np.arange(1, 10)              # 9 restaurant IDs
print(ids.reshape(3, 3))            # 3 cards in each row, in reading order

ids = np.arange(1, 11)              # 10 restaurant IDs
print(ids.reshape(-1, 2))           # a phone: 2 cards in a row, NumPy works out 5 rows
print(ids.reshape(-1, 5))           # a tablet: 5 cards in a row, NumPy works out 2 rows
# ids.reshape(3, 3)    ValueError: cannot reshape array of size 10 into shape (3,3)
# ids.reshape(-1, 3)   ValueError: cannot reshape array of size 10 into shape (3)

# A model in scikit-learn wants a table: one row per sample, one column per feature.
X = votes.reshape(-1, 1)
print(X.shape)          # (10, 1)

# .T and reshape give the same shape, not the same table
a = np.arange(1, 7).reshape(2, 3)
print(a.T)
print(a.reshape(3, 2))

# On the manager's sheet: .T keeps the votes together, reshape mixes votes and rupees
print(data.T[0, :4])
print(data.reshape(2, 10)[0, :4])
