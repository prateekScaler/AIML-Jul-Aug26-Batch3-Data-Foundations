# Class 2 · Practice: shape, ndim and size

29 questions on reading the shape of an array, from simple to medium. For each one: read the code, say the answer aloud, and only then open it. Every answer here was checked by running the code.

You need only `import numpy as np` at the top.

## Three rules that answer every question

1. **Count the brackets at the start.** `[` is 1D, `[[` is 2D, `[[[` is 3D. That number is `ndim`, and the shape has that many numbers.
2. **Count what each pair of brackets holds, from the outside in.** In `[[1, 2, 3], [4, 5, 6]]` the outer bracket holds 2 lists (2 rows) and each list holds 3 numbers (3 columns): shape `(2, 3)`. Rows first, because the outer bracket comes first.
3. **A number drops a direction. A slice keeps it.** `a[0]` loses the row direction. `a[0:1]` keeps it.

`size` is all the numbers in the shape multiplied together.

## Warm-up: one row

### 1. Three numbers

What does this print?

```python
a = np.array([5, 10, 15])
print(a.shape, a.ndim, a.size)
```

<details><summary>Answer</summary>

```text
(3,) 1 3
```

One pair of brackets, so one direction: `ndim` is 1. The bracket holds 3 numbers, so the shape is `(3,)` and the size is 3. The comma in `(3,)` only says "this is a tuple with one number".

</details>

### 2. One number

What does this print?

```python
a = np.array([7])
print(a.shape, a.ndim, a.size)
```

<details><summary>Answer</summary>

```text
(1,) 1 1
```

It is still an array with one direction. The bracket holds 1 number: shape `(1,)`, size 1. A single value inside brackets is a 1D array, not a plain number.

</details>

### 3. Ten restaurants

What does this print?

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])
print(votes.shape, votes.size)
```

<details><summary>Answer</summary>

```text
(10,) 10
```

One pair of brackets holding 10 numbers: one row of ten, `(10,)`. For a 1D array the size is the same number as the one in the shape.

</details>

### 4. Ratings with decimals

What does this print?

```python
ratings = np.array([4.5, 3.9, 4.1, 4.8, 3.2])
print(ratings.shape, ratings.dtype)
```

<details><summary>Answer</summary>

```text
(5,) float64
```

Five numbers in one row: `(5,)`. The kind of value does not change the shape. Decimals change the dtype (`float64`), not the count.

</details>

### 5. Text

What does this print?

```python
cities = np.array(["Jaipur", "Pune", "Indore", "Kochi"])
print(cities.shape, cities.size)
```

<details><summary>Answer</summary>

```text
(4,) 4
```

Four strings in one row: `(4,)`. The length of each string does not matter. One string is one element, however long it is. The letters are counted in the dtype (`<U6`), not in the shape.

</details>

### 6. A slice of a row

What does this print?

```python
a = np.array([10, 20, 30, 40, 50, 60])
print(a[1:4])
print(a[1:4].shape)
```

<details><summary>Answer</summary>

```text
[20 30 40]
(3,)
```

Start at position 1, stop before position 4: positions 1, 2 and 3. Three elements, still one row: `(3,)`. A slice of a 1D array is a 1D array.

</details>

## Core: rows and columns

### 7. Three lists of two

What does this print?

```python
a = np.array([[1, 2],
              [3, 4],
              [5, 6]])
print(a.shape, a.ndim, a.size)
```

<details><summary>Answer</summary>

```text
(3, 2) 2 6
```

Count what each pair of brackets holds, from the outside in. The outer bracket holds 3 lists: 3 rows. Each list holds 2 numbers: 2 columns. Shape `(3, 2)`. Two brackets at the start, so `ndim` is 2. Size is 3 × 2 = 6.

</details>

### 8. One row, but two brackets

What does this print?

```python
a = np.array([[1, 2, 3, 4, 5]])
print(a.shape, a.ndim)
```

<details><summary>Answer</summary>

```text
(1, 5) 2
```

Two brackets at the start, so it is 2D. The outer bracket holds 1 list: 1 row. That list holds 5 numbers: 5 columns. Shape `(1, 5)`. It looks like a row of five, but it is a table with one row.

</details>

### 9. One column

What does this print?

```python
a = np.array([[1],
              [2],
              [3]])
print(a.shape)
```

<details><summary>Answer</summary>

```text
(3, 1)
```

The outer bracket holds 3 lists: 3 rows. Each list holds 1 number: 1 column. Shape `(3, 1)`: a table with three rows and one column.

</details>

### 10. Same numbers, different brackets

What does this print?

```python
a = np.array([1, 2, 3])
b = np.array([[1, 2, 3]])
print(a.shape, b.shape)
```

<details><summary>Answer</summary>

```text
(3,) (1, 3)
```

`a` has one bracket: a row of 3, shape `(3,)`. `b` has two brackets: a table with 1 row and 3 columns, shape `(1, 3)`. The numbers are the same. The brackets decide the shape.

</details>

### 11. Read it backwards

What does this print?

```python
a = np.array([[4, 8, 15, 16, 23, 42],
              [1, 1, 2, 3, 5, 8],
              [9, 9, 9, 9, 9, 9],
              [0, 2, 4, 6, 8, 10]])
print(a.shape, a.size)
```

<details><summary>Answer</summary>

```text
(4, 6) 24
```

The outer bracket holds 4 lists: 4 rows. Each list holds 6 numbers: 6 columns. Shape `(4, 6)`. Size is rows × columns: 4 × 6 = 24.

</details>

### 12. Rows of different lengths

What happens?

```python
a = np.array([[1, 2, 3],
              [4, 5]])
```

<details><summary>Answer</summary>

```text
ValueError: setting an array element with a sequence.
```

An error. The first list holds 3 numbers and the second holds 2. A 2D array is a full rectangle: every row must have the same number of columns. NumPy cannot give this a shape, so it refuses. (Older NumPy versions gave a warning; from version 1.24 it is an error.)

</details>

### 13. Forcing rows of different lengths

What does this print?

```python
a = np.array([[1, 2, 3], [4, 5]], dtype=object)
print(a.shape)
print(a * 2)
```

<details><summary>Answer</summary>

```text
(2,)
[list([1, 2, 3, 1, 2, 3]) list([4, 5, 4, 5])]
```

`dtype=object` tells NumPy to store anything. So it stores two Python lists, one in each position: the shape is `(2,)`, a row of 2 things, not a table. And `* 2` now repeats each list, the way a list does. You keep the brackets but lose the fast maths. Use a full rectangle instead.

</details>

### 14. Transpose a table

What does this print?

```python
a = np.array([[1, 2, 3, 4, 5],
              [6, 7, 8, 9, 10]])
print(a.shape, a.T.shape)
```

<details><summary>Answer</summary>

```text
(2, 5) (5, 2)
```

`a` has 2 rows and 5 columns. `.T` turns every row into a column, so the two numbers swap: `(5, 2)`. The size stays 10.

</details>

### 15. Transpose a single row

What does this print?

```python
a = np.array([1, 2, 3, 4])
print(a.shape, a.T.shape)
```

<details><summary>Answer</summary>

```text
(4,) (4,)
```

No change. A 1D array has only one direction, so there is nothing to swap. `.T` matters only when there are two or more directions.

</details>

### 16. Stack, then transpose

What does this print?

```python
votes = np.array([775, 787, 918, 88])
costs = np.array([800, 800, 800, 300])
d = np.array([votes, costs])
print(d.shape, d.T.shape)
```

<details><summary>Answer</summary>

```text
(2, 4) (4, 2)
```

The outer bracket holds 2 arrays: 2 rows (one of votes, one of costs). Each holds 4 numbers: 4 columns. Shape `(2, 4)`. After `.T`: 4 rows and 2 columns, one row for each restaurant.

</details>

## Core: the shape after indexing and slicing

### 17. One row out

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a.shape)
print(a[0])
print(a[0].shape)
```

<details><summary>Answer</summary>

```text
(3, 4)
[10 11 12 13]
(4,)
```

`a` has 3 rows and 4 columns. `a[0]` uses a **number** for the row. A number picks one row and drops that direction, so only the 4 values come back: `(4,)`.

</details>

### 18. One column out

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a[:, 1])
print(a[:, 1].shape)
```

<details><summary>Answer</summary>

```text
[11 21 31]
(3,)
```

Every row, column number 1. The number drops the column direction. What is left is one value from each of the 3 rows: `(3,)`. It is not `(3, 1)`: one bracket, 1D.

</details>

### 19. A slice of rows

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a[1:3].shape)
```

<details><summary>Answer</summary>

```text
(2, 4)
```

`1:3` is a **slice**: rows 1 and 2 (3 is left out). A slice keeps the direction. So there are still two directions: 2 rows and all 4 columns, `(2, 4)`.

</details>

### 20. A slice of columns

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a[:, 1:3])
print(a[:, 1:3].shape)
```

<details><summary>Answer</summary>

```text
[[11 12]
 [21 22]
 [31 32]]
(3, 2)
```

All 3 rows, and columns 1 and 2. Both parts are slices, so both directions stay: `(3, 2)`.

</details>

### 21. A number or a slice

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a[0].shape, a[0:1].shape)
```

<details><summary>Answer</summary>

```text
(4,) (1, 4)
```

`a[0]` uses a number: the row direction is dropped, `(4,)`. `a[0:1]` uses a slice that holds one row: the row direction is kept, `(1, 4)`. The values are the same. **A number drops a direction. A slice keeps it.**

</details>

### 22. A corner

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a[:2, :2])
print(a[:2, :2].shape)
```

<details><summary>Answer</summary>

```text
[[10 11]
 [20 21]]
(2, 2)
```

Rows 0 and 1, columns 0 and 1: the top-left corner. Two slices, so two directions: `(2, 2)`.

</details>

### 23. One cell

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a[1, 2])
print(a[1, 2].shape)
```

<details><summary>Answer</summary>

```text
22
()
```

Row 1, column 2: the value 22. Two numbers were used, so both directions are dropped. Nothing is left in the shape: `()`. It is a single value, not a row.

</details>

### 24. The last column, two ways

What does this print?

```python
a = np.array([[10, 11, 12, 13],
              [20, 21, 22, 23],
              [30, 31, 32, 33]])
print(a[:, 3])
print(a[:, 3].shape)
print(a[:, 3:4].shape)
```

<details><summary>Answer</summary>

```text
[13 23 33]
(3,)
(3, 1)
```

Column 3 is the last column: 13, 23, 33. With the **number** `3`, the column direction is dropped: `(3,)`. With the **slice** `3:4`, which holds the same one column, the direction is kept: 3 rows and 1 column, `(3, 1)`.

</details>

### 25. Many positions at once

What does this print?

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])
picked = votes[[0, 2, 4]]
print(picked)
print(picked.shape)
```

<details><summary>Answer</summary>

```text
[775 918 166]
(3,)
```

A list of positions inside the brackets picks those elements: positions 0, 2 and 4. Three positions were asked for, so three values come back: `(3,)`. The shape follows the list of positions, not the original array.

</details>

## Stretch: three directions

### 26. Brackets inside brackets

What does this print?

```python
a = np.array([[[1, 2], [3, 4]],
              [[5, 6], [7, 8]]])
print(a.shape, a.ndim, a.size)
```

<details><summary>Answer</summary>

```text
(2, 2, 2) 3 8
```

Three brackets at the start: 3D. The outer bracket holds 2 blocks. Each block holds 2 rows. Each row holds 2 numbers. Shape `(2, 2, 2)`, size 2 × 2 × 2 = 8.

</details>

### 27. Two blocks of three rows

What does this print?

```python
a = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
              [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]])
print(a.shape, a.size)
```

<details><summary>Answer</summary>

```text
(2, 3, 4) 24
```

Outside in: the outer bracket holds 2 blocks. Each block holds 3 rows. Each row holds 4 numbers. Shape `(2, 3, 4)`. Size is 2 × 3 × 4 = 24.

</details>

### 28. One block out

What does this print?

```python
a = np.array([[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
              [[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2]]])
print(a[0].shape, a[0, 1].shape, a[0, 1, 2])
```

<details><summary>Answer</summary>

```text
(3, 4) (4,) 7
```

Each number you use drops one direction, from the left. `a[0]` is block 0: `(3, 4)`. `a[0, 1]` is row 1 of that block: `(4,)`. `a[0, 1, 2]` is one value: 7.

</details>

### 29. A colour photo

What does this print?

```python
img = np.zeros((427, 640, 3))   # a colour photo is stored in this shape
print(img.shape, img.size)
print(img[0].shape, img[:, :, 0].shape, img[100:300, 200:500].shape)
```

<details><summary>Answer</summary>

```text
(427, 640, 3) 819840
(640, 3) (427, 640) (200, 300, 3)
```

A colour photo is a 3D array: 427 rows of dots, 640 columns, and 3 colour values (red, green, blue) for every dot. 427 × 640 × 3 = 819,840 numbers. `img[0]` is the top row of dots: 640 dots with 3 values each. `img[:, :, 0]` keeps every row and column but only the red value: `(427, 640)`. The crop keeps rows 100 to 299 (200 rows) and columns 200 to 499 (300 columns), and all 3 colours. (`np.zeros` makes an array of that shape filled with zeros.)

</details>

## Make your own

Ask the AI assistant in Colab: *"Give me 5 NumPy arrays written with brackets. Do not show their shapes."* Write the shape of each on paper, then check with `.shape`. If the assistant and NumPy disagree, NumPy is right.
