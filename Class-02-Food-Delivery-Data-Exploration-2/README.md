# Class 2 · Food Delivery Data Exploration and Analysis 2

The second class of Data Foundations. It starts with five recap questions on Class 1, and then finishes the NumPy basics that Class 1 did not reach: `dtype`, shape, type conversion, indexing and slicing. The data is the same 10 restaurants: votes and the average cost for two.

**By the end of this class you can:**

- Read an array's `dtype`, `shape`, `ndim` and `size`.
- Convert types with `astype()`, and clean text before converting.
- Pick elements with indexing and slicing, in 1D and 2D.
- Say why a 2D shape is (rows, columns), and why every row needs the same length.

## How to use this folder

| File | What it is |
|---|---|
| [`index.html`](index.html) | The interactive class page. **Step by step** shows a little at a time (press → or Next). **Show everything** gives the whole class on one page. |
| [`code/`](code/) | Every example from the class as a runnable file, in order. |
| [`PRACTICE.md`](PRACTICE.md) | 29 questions on shape, ndim and size, simple to medium, each with an explained answer. |
| [`Data-Foundations-Class-02-Food-Delivery-Data-Exploration-2.ipynb`](Data-Foundations-Class-02-Food-Delivery-Data-Exploration-2.ipynb) | The same examples as one notebook. Upload it to [Google Colab](https://colab.research.google.com/) and run the cells, nothing to install. |

Run the code in Google Colab, or on your own machine after `pip install numpy`.

---

## Recap of Class 1, in five lines

- A rule that a machine finds from past data is **machine learning**. A rule that a person writes is a normal program.
- **EDA** comes first: understand and clean the data, then calculate. `float("1.200")` gives `1.2` with no error.
- A list times 2 is repeated. An array times 2 is doubled, with no loop: **vectorization**.
- NumPy is fast because the values sit side by side, share one type, and the loop runs in compiled C.
- **One type only.** `np.array([7, 2.5, True])` stores three floats: `[7.  2.5 1. ]`. True becomes 1 and False becomes 0: `np.array([True, False, 6])` stores `[1 0 6]`.

## 1. dtype

**dtype** means **data type**: the one type that every element of an array has. Read it with `array.dtype`.

| dtype | What the array holds |
|---|---|
| `int64` | whole numbers, 64 bits each |
| `float64` | decimal numbers, 64 bits each |
| `bool` | True or False |
| `<U32` | text, up to 32 characters each |

Read `<U32` in three parts: `<` is the byte order (how the bytes are stored; ignore it for now), `U` is Unicode, which means text, and `32` is the most characters one element can hold. `np.array(["abc", "de"])` gets `<U3`: the longest string has 3 characters.

**Why 32, when "3.5" has only 3 characters?** The number after `U` is the room kept for each element, not the length of your data. NumPy does not measure `3.5`; it keeps room for the longest float there could be, written as text. A float can be long: `-1.7976931348623157e+308` is 24 characters, and NumPy keeps 32 to be safe.

| You write | dtype | Why |
|---|---|---|
| `np.array(["Akash", "Ria"])` | `<U5` | only text: the longest string, 5 |
| `np.array([1, "Akash"])` | `<U21` | room for any integer: up to 20 digits and a sign |
| `np.array([3.5, "Akash"])` | `<U32` | room for any float |
| `np.array(["Ria", True])` | `<U5` | room for any Boolean: "False" has 5 characters |

**Why `<U5` and not `<U4` for `["Ria", True]`?** "True" has 4 characters, but NumPy does not look at the value. It looks at the type. A Boolean can also be False, and "False" has 5 characters, so NumPy keeps room for 5. If a string is longer, the string wins: `np.array(["Bengaluru", True])` gets `<U9`.

## 2. Shape and dimensions

A second array, `costs`, holds the average cost for two people. It is messy on purpose: each cost is text, with extra quotes inside.

```python
costs = np.array(["'800.0'", "'800.0'", "'800.0'", "'300.0'",
                  "'600.0'", "'600.0'", "'600.0'", "'700.0'",
                  "'550.0'", "'500.0'"])

print(votes.shape)   # (10,)  one direction, 10 elements
print(votes.ndim)    # 1
print(votes.size)    # 10

two_d_data = np.array([votes[:5], costs[:5]]).T   # .T turns rows into columns
print(two_d_data.shape)   # (5, 2): 5 restaurants, 2 attributes
```

**Transpose.** `.T` turns every row into a column. No value changes, only the layout.

```python
m = np.array([[1, 2, 3],
              [4, 5, 6]])
print(m.shape)     # (2, 3)
print(m.T)         # [[1 4]
                   #  [2 5]
                   #  [3 6]]
print(m.T.shape)   # (3, 2)
```

Why we need it: `np.array([votes[:5], costs[:5]])` stacks one row of votes and one row of costs, so a restaurant is split across two rows. After `.T` there is one row per restaurant, and its votes and its cost sit together. Data tables are kept this way: one row per record, one column per attribute.

Where else is transpose used? In maths, two matrices can be multiplied only when their shapes fit, and a transpose makes them fit. A photo is a grid of numbers, and transposing it flips the photo along its diagonal. In AI, a neural network multiplies large matrices at every layer, and the attention formula inside every LLM has a transpose in it.

**What the shape means.** `.shape` tells how many elements lie along each direction of the array. `votes` is one row, so it has one direction and one number: `(10,)`. A 2D array has two directions, down and across, so its shape has two numbers.

**Why rows come first.** Count what each pair of brackets holds, from the outside in. In `[[1, 2, 3], [4, 5, 6]]` the outer bracket holds 2 lists, stacked going down: 2 rows. Each list holds 3 numbers, going across: 3 columns. So the shape is `(2, 3)`. The same rule reads a 3D array: the outer bracket holds 2 blocks, each block holds 3 rows, each row holds 2 numbers, so `(2, 3, 2)`.

`.shape` gives (rows, columns), `.ndim` the number of dimensions, `.size` the total number of elements. For `votes` the shape `(10,)` and the size `10` look alike, but they are different things: an array with 2 rows and 3 columns has shape `(2, 3)`, size `6` (2 × 3) and ndim `2`. Shape describes the layout; size counts the elements. A shortcut: count the opening square brackets. `[` is 1D, `[[` is 2D, `[[[` is 3D.

Read a shape from the outside in:

```python
a = np.array([[1, 2, 3, 4],
              [5, 6, 7, 8],
              [9, 10, 11, 12]])
print(a.shape)   # (3, 4): 3 rows, 4 columns. Size 12.

b = np.array([[[1, 2], [3, 4], [5, 6]],
              [[7, 8], [9, 10], [11, 12]]])
print(b.shape)   # (2, 3, 2): 2 blocks, 3 rows in each, 2 numbers in each row. Size 12, ndim 3.
```

**Rows of different lengths are not allowed.** A 2D array is a full rectangle: every row must have the same number of columns.

```python
np.array([[1, 2, 3], [4, 5]])
# ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape ...
```

You can force it with `dtype=object`, but then each row is a plain Python list stored inside the array. The shape is `(2,)`, not a table, and you lose the fast maths:

```python
a = np.array([[1, 2, 3], [4, 5]], dtype=object)
print(a.shape)    # (2,)
print(a * 2)      # [list([1, 2, 3, 1, 2, 3]) list([4, 5, 4, 5])]  the lists are repeated, not doubled
```

In `two_d_data` the votes print as `'775'`, in quotes. An array holds one type, the costs are strings, so NumPy quietly turned the votes into strings too.

## 3. Type conversion

Text does not add up: `"500" + "4"` is `"5004"`, while `500 + 4` is `504`. Numbers stored as text must be converted first.

```python
a = np.array(['1.2', '2.5', '3.6', '4.8'])
print(a.dtype)             # <U3: text, up to 3 characters each
a = a.astype(float)
print(a.dtype)             # float64

np.array(["abc", "xyz"]).astype(float)
# ValueError: could not convert string to float: np.str_('abc')

np.array([1.7, 2.8, 3.9]).astype(int)   # [1 2 3]: cut off, not rounded
```

Our costs fail in the same way, because the extra quotes are part of the text. Clean first, then convert:

```python
clean_costs = np.char.strip(costs, "'").astype(float)
print(clean_costs)            # [800. 800. 800. 300. 600. 600. 600. 700. 550. 500.]
print(clean_costs.mean())     # 625.0: the average cost for two
```

## 4. Indexing and slicing

```python
print(votes[0])     # 775   the first
print(votes[-1])    # 402   the last
print(votes[-2])    # 504   the second last
print(votes[[2, 3, 4, 1, 2, 2]])   # [918  88 166 787 918 918]  many positions at once

a = np.array([100, 200, 300, 400, 500])
print(a[1:4])       # [200 300 400]  starts at 1, stops before 4
```

Counting from the end starts at -1, because -0 is the same as 0. In a slice `start:end`, the end is left out (like a hotel: check in on the 1st, check out on the 4th, and you stay nights 1, 2 and 3), so to reach position 2 you write `0:3`. Pagination is slicing: page 1 is `a[0:10]`, page 2 is `a[10:20]`. A slice that goes past the end does not fail; it stops at the end.

In 2D the order is **row, then column**:

| You write | You get |
|---|---|
| `two_d_data[1, 0]` | row 1, column 0: `787` |
| `two_d_data[:3, :]` | the first 3 rows, all columns |
| `two_d_data[:, 0]` | every row, column 0: all the votes |
| `arr[0:2, 1:3]` | rows 0 and 1, columns 1 and 2 |

One column can come back in two shapes. A slice keeps a direction; a number drops it:

```python
arr = np.array([[1, 2],
                [3, 4],
                [5, 6]])
print(arr[:, 0:1])        # [[1] [3] [5]]  a slice: two brackets, still 2D
print(arr[:, 0:1].shape)  # (3, 1)
print(arr[:, 0])          # [1 3 5]        a number: one bracket, 1D
print(arr[:, 0].shape)    # (3,)
```

Explore slicing with the [NumPy slicing visualizer](https://aiml-v-izs.vercel.app/v/fd-k7q2x9m4/slicing?embed=1).

## Recap

- `dtype` is the one type of an array. The number after `U` is the room kept for each element.
- `.shape`, `.ndim`, `.size` describe an array. `.T` swaps rows and columns.
- `astype()` converts types. Clean the text first: the average cost for two is ₹625.
- **Indexing and slicing:** start at 0, the end is left out, and in 2D it is row, then column. A number drops a direction; a slice keeps it.
- Rows of different lengths are not allowed. A 2D array is a full rectangle.

**Next class:** `np.arange`, slicing with a step, and filtering with masks.
