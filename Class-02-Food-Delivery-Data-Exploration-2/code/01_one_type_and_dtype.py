# A NumPy array holds one type only. Mixed values are all changed to the
# "highest" type: String > Float > Integer > Boolean.
import numpy as np

print(np.array([1, 2, 3.5]))            # all become floats
print(np.array([1, "Akash", 3.5]))      # all become strings
print(np.array([True, 6]))              # True becomes 1

# dtype = data type: the one type that every element has.
# <U32 reads as: U = Unicode text, 32 = up to 32 characters in each element.
# (The < is the byte order: how the bytes are stored. Ignore it for now.)
print(np.array([1, "Akash", 3.5]).dtype)   # <U32
print(np.array(["abc", "de"]).dtype)       # <U3: the longest string has 3 characters
print(np.array([1, 2]).dtype)              # int64
print(np.array([1.5]).dtype)               # float64

# Why 32? The number after U is the room kept for each element.
# NumPy keeps room for the longest value that type could hold, written as text.
print(np.array(["Akash", "Ria"]).dtype)    # <U5: only text, the longest string has 5 characters
print(np.array([1, "Akash"]).dtype)        # <U21: room for any integer (up to 20 digits and a sign)
print(np.array([3.5, "Akash"]).dtype)      # <U32: room for any float

# Why <U5 and not <U4? NumPy keeps room for any Boolean, and "False" has 5 characters.
print(np.array(["Ria", True]).dtype)         # <U5
print(np.array(["Bengaluru", True]).dtype)   # <U9: a longer string wins
