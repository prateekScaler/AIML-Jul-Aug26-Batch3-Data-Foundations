# Class 1 · Food Delivery Data Exploration and Analysis 1

The first class of Data Foundations. It answers four questions: why data matters, what machine learning is, why we explore and clean data before a machine sees it, and how NumPy lets us work with numbers fast. The data is a small set of 10 restaurants from a food delivery app: how many votes each got, and the average cost for two.

**By the end of this class you can:**

- Say what machine learning is, with an example.
- Explain what EDA is and why it comes first.
- Create a NumPy array and say why it is faster than a Python list.
- Read an array's `shape`, `ndim` and `size`.
- Convert types with `astype()`, and clean text before converting.
- Pick elements with indexing and slicing, in 1D and 2D.

**In the live class** we reached "One type only" (in section 4). Class 2 starts with a short recap and continues from `dtype`. The full page and notes are here, so you can read ahead.

## How to use this folder

| File | What it is |
|---|---|
| [`index.html`](index.html) | The interactive class page. **Step by step** shows a little at a time (press → or Next). **Show everything** gives the whole class on one page. |
| [`code/`](code/) | Every example from the class as a runnable file, in order. |
| [`Data-Foundations-Class-01-Food-Delivery-Data-Exploration-1.ipynb`](Data-Foundations-Class-01-Food-Delivery-Data-Exploration-1.ipynb) | The same examples as one notebook. Upload it to [Google Colab](https://colab.research.google.com/) and run the cells, nothing to install. |

Run the code in Google Colab, or on your own machine after `pip install numpy`.

---

## Why data matters

**Two biryani shops on the same street.** Same biryani, same price. Shop A cooks by guess. Shop B first looks at last month's orders, and learns three things: Friday night sells double, so it cooks more on Friday; most orders come from one area, so it keeps its riders there; ratings fall when delivery takes over 40 minutes, so it fixes that first. Data turns a guess into a decision.

**Every AI you have used was built from data.**

| The AI | The data it learned from |
|---|---|
| ChatGPT, Claude, Gemini | text written by people |
| Netflix and YouTube suggestions | what users watched |
| Delivery time on a food app | lakhs of past orders |
| A bank's loan model | past loans: who repaid |

No data, no AI. Bad data, bad AI.

**Where does a data scientist's time go?** In a survey of 2,360 data people, they estimated their own time like this ([Anaconda, State of Data Science 2020](https://know.anaconda.com/rs/387-XNW-688/images/Anaconda-SODS-Report-2020-Final.pdf)):

| Job | Share of time |
|---|---|
| **Loading and cleaning data** | **45%** |
| **Charts: visualising the data** | **21%** |
| Training the model | 12% |
| Choosing the model | 11% |
| Deploying the model | 11% |

The first two rows are this module: 66 of every 100 hours. The model itself is the smaller part.

**What this module gives you.** One journey, three times: a raw file, clean it, analyse it, draw it, reach a decision.

| Classes | The data | The tools |
|---|---|---|
| 1 to 5 | Restaurants on a food delivery app | NumPy, Pandas |
| 6 to 9 | Amazon sales | Pandas, Matplotlib, Seaborn |
| 10 | Sachin Tendulkar's ODI career | all of them |

After this module, you can take a raw file and a business question, and return an answer with a chart.

## 1. What is machine learning?

**Machine learning** is the ability of a machine to learn from experience, history and patterns, the way humans do.

- **A bank loan.** A bank can give one ₹20 lakh loan. Candidate 1 earns ₹5 lakh a month, Candidate 2 earns ₹5 lakh a year. Most people choose Candidate 1. Now two more facts: Candidate 1 owns a flat in Bengaluru and lives in London. Sounds familiar? It is a joke, but notice what you did: you used a case from the past to judge a new one. Banks do the same with millions of past cases. That is machine learning.
- **A salary.** 1 year of experience pays ₹10k, 2 years ₹20k, 3 years ₹30k. You can predict ₹60k for 6 years, because you found the pattern: salary = years × ₹10k. You did it from 3 records; a machine does it across 10 million.
- **Your mother.** She knows when you are lying, without proof. She has watched your tone and reactions for years and learned your patterns. More history, better guesses: a machine also needs a lot of data.
- **Netflix.** More than 250 million users watch, pause, rewind and drop shows. Each action is data. A user who takes a free month and cancels is a loss, because Netflix paid to get that user and earned nothing. So its real job is to keep you: it must guess what you will watch next, and it learns that from what millions of users did.

**Where is the line?** Ask who wrote the rule.

| | What goes in | What comes out |
|---|---|---|
| Normal program | data + a rule written by a person | answers |
| Machine learning | data + past answers | a rule, found by the machine |

A food delivery app sorts 2 lakh restaurants by rating, adds 5% GST to each bill, and gives ₹50 off to users who have not ordered for 30 days. None of these is machine learning: a person wrote each rule. Estimating the delivery time of a new order is machine learning: nobody can write that rule by hand (traffic, rain, the restaurant, the hour), so the machine finds it from lakhs of past orders. A lot of data is not machine learning. Using history is not machine learning. The rule coming from the data is.

Companies make better decisions when they learn patterns from data. But before a machine can learn, we must understand the data ourselves.

## 2. EDA and DAV

**EDA**, Exploratory Data Analysis, is exploring and analysing every feature of a dataset before taking a decision. Before a movie you check the rating, reviews, actors, type and price, not one factor alone.

**First decide what to measure.** Three movies are showing tonight. Which is the best?

| Movie | Rating | Ticket price | Length | Type |
|---|---|---|---|---|
| Movie A | 8.6 / 10 | ₹450 | 3h 05m | Thriller |
| Movie B | 7.8 / 10 | ₹250 | 2h 10m | Family comedy |
| Movie C | 6.9 / 10 | ₹150 | 2h 40m | Horror |

By rating, A wins. By price, C wins. By length, B wins. The data did not change; the measure you chose decided the answer. So the first job is to choose what to measure.

So EDA starts here: decide what to measure, then look at every measure before you choose.

EDA is needed for two more reasons:

1. **A machine has no common sense.** In a table with a serial number, experience and salary, the serial number also rises with salary. A machine may treat it as a cause. A person removes that column.
2. **Real data is messy.** `"10k"` must become `10000`, `"4.1/5"` must become `4.1`, and a cost stored as text must become a number before any maths.

**The same number can be written in different ways.**

| Where | How 1200.50 is written |
|---|---|
| India, US, UK | `1,200.50` |
| Germany, Spain, Italy | `1.200,50` |
| France | `1 200,50` |
| Switzerland | `1'200.50` |

A German price file lists a laptop at `1.200` euros. `float("1.200")` gives `1.2`, with no error: a 1,200 euro laptop became 1.2 euros. A wrong number with no error is the hardest bug to find. India has its own style too: `1,20,000` here is `120,000` in the US. Before any maths, ask where the data came from, then clean it into one format.

**EDA is the job. DAV is the toolkit.** Three movies can be explored by hand. Thousands of restaurants cannot. **DAV**, Data Analysis and Visualization, is the set of tools that do EDA on real data:

| What EDA asks us to do | The tool |
|---|---|
| Maths on many numbers: totals, averages | **NumPy** |
| Tables with many columns, cleaning text | **Pandas** |
| Charts | **Matplotlib** and **Seaborn** |

## 3. The dataset and the tools

You are a data analyst at a food delivery company. The practice dataset has 10 restaurants and two columns: `votes`, the number of customer ratings, and `costs`, the average cost for two in ₹. Your manager has three questions: which restaurants are most popular, are popular places more expensive, and what is the average cost for two people? This class answers the third, after cleaning the data: ₹625.

- **Google Colab** is a coding notebook in the browser. The code runs on Google's computers, the libraries are already installed, and a cell runs with Shift + Enter.
- `import numpy as np` brings in NumPy with the short name `np`, the name everyone uses.
- Colab has an AI assistant, Gemini: it can generate code, explain code and answer questions. Let it write and explain; you predict, run and verify.

## 4. NumPy arrays

**NumPy** is short for **Num**erical **Py**thon. Travis Oliphant made it in 2005; it is free and open source, and Pandas and most data libraries are built on it.

A list is not built for maths: `[4, 5, 6] * 2` gives `[4, 5, 6, 4, 5, 6]`, the list repeated.

```python
import numpy as np
votes = np.array([ 775,  787,  918,   88,  166,  286, 2556,  324,  504,  402])
print(type(votes))    # <class 'numpy.ndarray'>: n-dimensional array

print(votes * 2)       # [1550 1574 1836  176  332  572 5112  648 1008  804]
print(votes.sum(), votes.max(), votes.mean())   # 6806 2556 680.6
```

An array applies the operation to every element at once. This is **vectorization**: no loop.

**Why it is fast.** A list stores addresses, and its values are scattered in memory. An array stores the values themselves, one after the other, all of one type. Think of two tea rooms. In Room 1 the tea, sugar, milk and cups lie around the house: that is a list. In Room 2 they are on one shelf: that is an array. On one million numbers, doubling with an array is many times faster than with a list ([`code/03_why_numpy_is_faster.py`](code/03_why_numpy_is_faster.py)).

**Closer to the machine.** Languages sit at different levels. The same addition:

| Level | Looks like | |
|---|---|---|
| Python | `total = a + b` | easiest to write |
| C / C++ | `int total = a + b;` | |
| Assembly | `ADD AX, BX` | |
| Machine code | `00000001 11011000` | fastest to run; the only one the machine understands |

How big is the gap? For one billion loop steps, Rust takes 0.50 seconds and Python takes 74.42 seconds: watch the [animation by Benjamin Dicken](https://benjdd.com/languages/).

A Python loop does every step in Python: check the type, find the value, multiply. NumPy itself is written mostly in C. You write one line of Python, `votes * 2`, and the loop over the numbers runs in compiled C. NumPy is easy like Python and fast like C.

**One type only.** If you mix types, NumPy converts everything to the highest: String > Float > Integer > Boolean.

How does NumPy choose? It takes the type that can hold every value without losing any. `True` can be written as `1`, `1` as `1.0`, and `1.0` as `'1.0'`, but text such as `"two"` cannot become a number. So `np.array([1, "two", 3, 4, 5])` becomes all strings, even with four integers and one string. The first element and the majority do not matter.

```python
np.array([1, 2, 3.5])          # [1.  2.  3.5]        all floats
np.array([1, "Akash", 3.5])    # ['1' 'Akash' '3.5']  all strings
np.array([True, 6])            # [1 6]                True became 1
```

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

## 5. Shape and dimensions

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

In `two_d_data` the votes print as `'775'`, in quotes. An array holds one type, the costs are strings, so NumPy quietly turned the votes into strings too.

## 6. Type conversion

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

## 7. Indexing and slicing

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

One column of an (m, n) array has shape `(m,)`, not `(m, 1)`. A number drops a direction; a slice keeps it:

```python
arr = np.array([[1, 2],
                [3, 4],
                [5, 6]])
print(arr[:, 0])          # [1 3 5]        one bracket: 1D
print(arr[:, 0].shape)    # (3,)
print(arr[:, 0:1])        # [[1] [3] [5]]  two brackets: 2D
print(arr[:, 0:1].shape)  # (3, 1)
```

Explore slicing with the [NumPy slicing visualizer](https://aiml-v-izs.vercel.app/v/fd-k7q2x9m4/slicing?embed=1).

## Recap

- **Machine learning** learns patterns from history. **EDA** is how we understand and clean the data first.
- **NumPy arrays** do maths on every element at once, fast, because values sit side by side and share one type.
- **One type only:** String > Float > Integer > Boolean.
- `.shape`, `.ndim`, `.size` describe an array.
- `astype()` converts types. Clean the text first: the average cost for two is ₹625.
- **Indexing and slicing:** start at 0, `-1` is the last, the end of a slice is left out, and in 2D it is row, then column.
