# Why NumPy: the same job on one million numbers, with a list and with an array.
# The times change from run to run and from machine to machine.
import time
import numpy as np

n = 1_000_000
numbers_list = list(range(n))
numbers_array = np.arange(n)

start = time.time()
doubled_list = [x * 2 for x in numbers_list]
list_seconds = time.time() - start

start = time.time()
doubled_array = numbers_array * 2
array_seconds = time.time() - start

print(f"List:  {list_seconds:.4f} seconds")
print(f"Array: {array_seconds:.4f} seconds")
times = list_seconds / array_seconds
print(f"The array was about {times:.0f} times faster")
