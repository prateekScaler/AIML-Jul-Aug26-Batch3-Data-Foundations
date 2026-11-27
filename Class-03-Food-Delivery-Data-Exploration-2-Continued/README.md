# Class 3 · Food Delivery Data Exploration and Analysis 2 Continued

The third class of Data Foundations. It starts with three recap questions on Class 2, makes numbers with `np.arange`, slices with a step, and filters with masks. Then it builds a table from two arrays and asks it questions: totals, averages, if-else labels and rankings. The data is the same 10 restaurants: votes and the average cost for two.

This class was taught by **Shubham Singh**. His class notes are in [`Class-Notes-Shubham-Singh.pdf`](Class-Notes-Shubham-Singh.pdf). The class page, notes and code in this folder cover the same topics, with more examples and questions.

We are the data team of a food delivery app. Every tool today answers a question the business asks:

| Who asks | The question | The tool |
|---|---|---|
| The app team | Customer IDs, and filter buttons for cost and rating | `np.arange`, slicing with a step |
| The manager | Are popular places more expensive? | a mask |
| Marketing | Which places are popular **and** affordable? | two conditions, `&` |
| The manager | One sheet and one summary: total votes, average cost | `column_stack`, `sum`, `mean`, `axis` |
| Operations | Can a restaurant promise a delivery time? | `std` |
| The product team | Which cards get a "Popular" badge? | `np.where` |
| The manager | Which places are most popular, and what do they cost? | `np.argsort` |

The answers are collected at the end, in [The business questions, answered](#the-business-questions-answered).

**By the end of this class you can:**

- Make numbers with `np.arange(start, end, step)`, slice with a step, and reverse an array.
- Filter an array with a mask, and join two conditions with `&`.
- Build a table with `np.column_stack`, and change its shape with `reshape` and `-1`.
- Summarise with `sum`, `mean`, `max`, `min` and `std`, per column or per row with `axis`.
- Ask yes-or-no questions with `np.any` and `np.all`, and label every element with `np.where`.
- Sort with `np.sort`, and keep parallel arrays together with `np.argsort`.

## How to use this folder

| File | What it is |
|---|---|
| [`index.html`](index.html) | The interactive class page. **Step by step** shows a little at a time (press → or Next). **Show everything** gives the whole class on one page. |
| [`Class-Notes-Shubham-Singh.pdf`](Class-Notes-Shubham-Singh.pdf) | Shubham Singh's notes from the live class (3 pages). |
| [`code/`](code/) | Every example from the class as a runnable file, in order. |
| [`PRACTICE.md`](PRACTICE.md) | 26 practice questions on new business data, from warm-up to a Challenge band, each with an explained answer. |
| [`Data-Foundations-Class-03-Food-Delivery-Data-Exploration-2-Continued.ipynb`](Data-Foundations-Class-03-Food-Delivery-Data-Exploration-2-Continued.ipynb) | The same examples as one notebook. Upload it to [Google Colab](https://colab.research.google.com/) and run the cells, nothing to install. |

Run the code in Google Colab, or on your own machine after `pip install numpy scikit-learn matplotlib`.

---

## Recap of Class 2, in three lines

- Read a shape from the outside in: `[[1, 2, 3], [4, 5, 6]]` has 2 rows and 3 columns, `(2, 3)`.
- A slice keeps a direction, a number drops it: `arr[:, 0:1]` is `(3, 1)`, `arr[:, 0]` is `(3,)`.
- Text is not a number. Clean, then convert: `np.char.strip(costs, "'").astype(float)`. In this class that clean array is `clean_costs`.

## 1. np.arange: numbers in a row

*The app team: the app launches in a new city with its first 1,000 customers, and each needs an ID.*

`np.arange(start, end, step)` makes the numbers from `start`, jumping by `step`, and stops before `end`. The end is left out, as in a slice. The step is 1 when you leave it out.

```python
ids = np.arange(1, 1001)     # IDs for 1000 customers
print(ids[:5])               # [1 2 3 4 5]
print(ids[-3:])              # [ 998  999 1000]
print(ids.size)              # 1000

print(np.arange(1, 1000)[-1])    # 999: the end is left out
print(np.arange(100, 1000, 100))  # [100 200 300 400 500 600 700 800 900]  cost-for-two filter buttons
print(np.arange(3, 5, 0.5))       # [3.  3.5 4.  4.5]  rating filter buttons
```

Order IDs, ticket numbers and filter buttons are made this way: one line, no typing, none missing. `np.arange(100, 900, 100)` stops at 800, because 900 is left out. A decimal step works in `np.arange`; Python's own `range(3, 5, 0.5)` gives a TypeError.

**Why the business needs this:** change one number, the step, and every filter button on the app changes with it.

## 2. Slicing with a step

*The quality team cannot visit all 10 restaurants this week: they check every second one on the list. And the "Your orders" screen must show the newest order first.*

A slice can take a third number: `array[start : end : step]`. The step is the size of the jump, not a position.

```python
a = np.arange(10)    # [0 1 2 3 4 5 6 7 8 9]
print(a[::2])        # [0 2 4 6 8]             every second element
print(a[::-1])       # [9 8 7 6 5 4 3 2 1 0]   reversed
print(a[::-2])       # [9 7 5 3 1]             from the end, every second element
print(a[::-3])       # [9 6 3 0]
print(a[1:8:3])      # [1 4 7]                 start at 1, jump by 3, stop before 8
```

A negative step starts at the end and walks to the left. On our data, `votes[::2]` gives `[ 775  918  166 2556  504]`, the votes of the five places to visit; `[::-1]` turns an oldest-first list of orders into newest first.

**Why the business needs this:** a fixed rule, every second place, picks the visits, so nobody picks favourites.

## 3. Data Detective: a photo is an array

*The app team: every restaurant card shows a photo, and the app needs small thumbnails of all of them.* To Python, a photo is a NumPy array of numbers.

```python
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

img = load_sample_image("china.jpg")
print(type(img))               # <class 'numpy.ndarray'>
print(img.shape, img.dtype)    # (427, 640, 3) uint8
plt.imshow(img)                # draws the array as a picture
```

The shape says 427 rows, 640 columns, and 3 numbers for every dot: red, green and blue. `uint8` means whole numbers from 0 to 255. The top-left dot, `img[0, 0]`, is `[174 201 231]`.

So slicing a photo changes the picture:

| You write | What you see |
|---|---|
| `plt.imshow(img[100:300, 200:500])` | a crop: rows 100 to 299, columns 200 to 499 |
| `plt.imshow(img[::-1])` | the photo upside down: the rows are reversed |
| `plt.imshow(img[:, ::-1])` | a mirror image: the columns are reversed |
| `plt.imshow(img[::4, ::4])` | a smaller photo: every 4th row and column |
| `plt.imshow(255 - img)` | the negative: every number taken away from 255 |

Crop, flip, mirror, shrink: all of it is slicing. A thumbnail is `img[::4, ::4]`, from one line. The photo comes with scikit-learn: [load_sample_image](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_sample_image.html).

## 4. Filtering with masks

*The product team wants a "Popular" filter on the listing: only the places with 500 votes or more.* Shopping and food apps are full of filters, for example "rating 4 and above" or "cost under ₹200". Every filter starts with a question for every element.

```python
print(votes >= 500)
# [ True  True  True False False False  True False  True False]
```

This array of True and False is a **mask**. It has the same length and the same order as the array: answer number 4 belongs to value number 4. Why "mask"? A mask covers some things and lets others show. Think of a sheet of paper with holes, laid over the array: True is a hole and the value shows through, False is covered.

Now think of a gatekeeper: True, you may enter; False, you stay out. Put the mask inside the brackets, and only the elements where it is True come through:

```python
print(votes[votes >= 500])     # [ 775  787  918 2556  504]
print((votes >= 500).sum())    # 5: how many passed
```

True counts as 1 and False as 0, so the sum of a mask counts how many passed. No loop and no `if`: one line filters the whole array.

**How to read `votes[votes >= 500]`:** from the inside out, in two steps. Inside the brackets, `votes >= 500` asks every element the question and makes the mask. Outside, `votes[ mask ]` keeps the values where the mask is True. It is the same as two lines:

```python
mask = votes >= 500
print(votes[mask])     # [ 775  787  918 2556  504]
```

Say it aloud: "from votes, the ones where votes is 500 or more". See it move in the [Boolean masking visualizer](https://aiml-v-izs.vercel.app/v/fd-r8w5n3p2/boolean-mask?embed=1).

**A mask from one array can filter another.** The mask knows nothing about restaurants. It works by **position**: the True at position 6 picks the element at position 6. This is called **boolean indexing**. `votes` and `clean_costs` are **parallel arrays**: the same position is the same restaurant.

```python
clean_costs = np.char.strip(costs, "'").astype(float)
print(clean_costs[votes >= 500])           # [800. 800. 800. 600. 550.]
print(clean_costs[votes >= 500].mean())    # 710.0  popular places
print(clean_costs[votes < 500].mean())     # 540.0  the rest
```

It works only when the two arrays match. A mask with 5 values on an array of 10 gives `IndexError: boolean index did not match indexed array`. Worse, the same length in a different order gives no error, only a wrong answer. Keep parallel arrays in the same order.

This answers the manager's question from Class 1: in this sample of 10 restaurants, popular places cost more (₹710 for two, against ₹540).

**Why the business needs this:** a "Popular" list is not a budget list, so the app should show the cost next to it.

**Two conditions at once.** *Marketing wants a "Popular and affordable" collection: over 500 votes and under ₹700 for two.* Put each condition in round brackets and join them with `&`:

```python
mask = (votes > 500) & (clean_costs < 700)
print(votes[mask])          # [2556  504]
print(clean_costs[mask])    # [600. 550.]
```

`&` joins two masks element by element: True only where both are True. The word `and` fails here with a ValueError, because it wants one True or False, not ten. Without the round brackets, `&` runs before `>` and you get a TypeError. Two places make the collection; every extra condition makes the list shorter.

**The same idea in another business.** A fitness band company asks: do happy users walk more? The band records the mood and the step count of each user:

```python
mood = np.array(["Happy", "Sad", "Happy", "Sad", "Happy", "Happy", "Sad", "Happy"])
steps = np.array([9200, 4100, 10500, 5200, 8700, 11000, 3900, 9600])
print(steps[mood == "Happy"].mean())   # 9800.0
print(steps[mood == "Sad"].mean())     # 4400.0
```

The mask comes from `mood` and filters `steps`. This is sample data, not a study.

## 5. Two arrays, one table: `np.column_stack`

*The manager: "I do not want two lists. Give me one sheet: one row per restaurant."* `votes` and `clean_costs` are parallel: position 0 is the same restaurant. `np.column_stack` makes each array a column of one table, so each row is one restaurant.

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])
clean_costs = np.array([800., 800., 800., 300., 600., 600., 600., 700., 550., 500.])

data = np.column_stack((votes, clean_costs))
print(data.shape)    # (10, 2)
print(data[:3])
# [[775. 800.]
#  [787. 800.]
#  [918. 800.]]
```

Mind the double round brackets: the arrays go in as one tuple. The votes became floats: an array holds one type, and float can hold both columns. `np.array([votes, clean_costs])` would give `(2, 10)`, with each array as a row.

**Why the business needs this:** one row is one restaurant, so a later filter or sort can never pull a place apart from its cost.

## 6. `reshape`: same numbers, new rows and columns

*The app team: the home screen shows restaurant cards in a grid, 3 in each row.* `reshape` refills the same numbers into new rows and columns, in reading order: the first row, then the next.

```python
ids = np.arange(1, 10)       # 9 restaurant IDs
print(ids.reshape(3, 3))
# [[1 2 3]
#  [4 5 6]
#  [7 8 9]]

np.arange(1, 11).reshape(3, 3)    # a 10th restaurant joined
# ValueError: cannot reshape array of size 10 into shape (3,3)
```

**Rows × columns must equal the size.** 10 numbers fit `(2, 5)`, `(5, 2)`, `(10, 1)` and `(1, 10)`, not `(3, 3)` or `(3, 4)`. NumPy never drops or invents data: it refuses. That is good news: no restaurant silently disappears from the screen.

**`-1` lets NumPy work out one side.** A phone shows 2 cards in a row, a tablet 5. On 10 IDs, `reshape(-1, 2)` gives `(5, 2)` and `reshape(-1, 5)` gives `(2, 5)`: nobody counts the rows by hand. `reshape(-1, 3)` still fails, because 10 ÷ 3 is not whole, and only one `-1` is allowed.

**Where you will meet it.** *The data science team asks: can a model predict the cost for two from the votes?* scikit-learn, the machine learning library, wants a table: one row per sample, one column per feature. Give it a 1D array and it says:

```text
ValueError: Expected 2D array, got 1D array instead:
array=[ 775  787  918   88  166  286 2556  324  504  402].
Reshape your data either using array.reshape(-1, 1) if your data has a single feature or array.reshape(1, -1) if it contains a single sample.
```

`votes.reshape(-1, 1)` has shape `(10, 1)`: one column, as many rows as needed, and the model accepts it.

**`reshape` is not `.T`.** For `a = np.arange(1, 7).reshape(2, 3)`, both `a.T` and `a.reshape(3, 2)` have shape `(3, 2)`, but they are different tables:

```python
print(a.T)              # [[1 4] [2 5] [3 6]]   each row became a column
print(a.reshape(3, 2))  # [[1 2] [3 4] [5 6]]   refilled in reading order
```

On the manager's sheet the difference is real: `data.T[0, :4]` is the votes, `[775. 787. 918.  88.]`, but `data.reshape(2, 10)[0, :4]` is `[775. 800. 787. 800.]`, votes and rupees mixed, with no error. To swap rows and columns, use `.T`. `reshape` can mix up your data, so use it with care. In Colab, a line ending in `#@param` becomes a slider or a box, which makes a quick reshape playground (see the class page).

## 7. Aggregates and `axis`

*The manager wants one slide for the weekly review.* Aggregates turn many numbers into one answer.

```python
print(data[:, 0].sum())     # 6806.0   total votes
print(data[:, 1].mean())    # 625.0    average cost for two
print(data[:, 0].max())     # 2556.0   most votes
print(data[:, 1].min())     # 300.0    cheapest
```

`np.sum(x)` and `x.sum()` do the same thing, and so do `mean`, `max`, `min` and `std`.

**Standard deviation** measures how far the values sit from the average. *Operations asks: can a restaurant show "Delivery in 30 min" on its card?* Two restaurants both average 30 minutes: A took 28, 30, 32 (std 1.63) and B took 10, 30, 50 (std 16.33). A can make the promise; for B the average describes none of its orders well. A delivery promise needs a small average **and** a small spread.

```python
a = np.array([28, 30, 32])
b = np.array([10, 30, 50])
print(a.mean(), b.mean())                    # 30.0 30.0
print(a.std().round(2), b.std().round(2))    # 1.63 16.33
```

Real: quick-commerce apps in India built their brand on "10-minute delivery". Blinkit dropped the claim from its ads in January 2026, after the labour ministry raised rider-safety concerns ([Inc42](https://inc42.com/buzz/blinkit-drops-10-minute-delivery-claim-after-govt-push/)). A promise is a business decision, not just a number.

**`axis` on a table.** *Operations plans riders for the weekend: orders of 3 restaurants (rows) on Friday, Saturday and Sunday (columns).* The axis you name is the one that disappears.

```python
orders = np.array([[20, 35, 30],
                   [12, 18, 15],
                   [25, 40, 32]])
print(np.sum(orders))           # 227         one number
print(np.sum(orders, axis=0))   # [57 93 77]  down the rows: one total per day
print(np.sum(orders, axis=1))   # [85 45 97]  across the columns: one total per restaurant
```

Saturday is the busiest day (93 orders): plan the most riders for Saturday. Restaurant 1 had only 45 orders and may need an offer.

In Excel terms, `axis=0` is the total row at the bottom and `axis=1` the total column on the right. A table of shape `(3, 4)` gives shape `(4,)` with `axis=0` and `(3,)` with `axis=1`.

On the restaurant table, `data.mean(axis=0)` gives `[680.6 625. ]`: average votes and average cost in one line. `data.sum(axis=1)` also runs, but votes plus rupees means nothing. NumPy checks the shape, not the meaning. And `data.max(axis=0)` gives `[2556. 800.]`: the top of each column, which do not come from one restaurant.

A mask and an aggregate together: *a restaurant owner asks whether the ₹800 places get more votes than the average place.* `votes[clean_costs == 800].mean()` is about 826.7, against 680.6: yes, though 3 places are a hint, not proof.

## 8. `np.any`, `np.all` and `np.where`

```python
print(np.any(clean_costs > 3000))   # False: is at least one True?
print(np.all(clean_costs < 5000))   # True:  is every one True?
```

*The manager: before the weekly slide goes out, check the data.* Each gives one True or False for the whole array, unless you give an `axis`. A "Top rated" badge that needs every one of the last 3 months above 4.2 is `all`: one rating of 4.1 makes it False. These checks stop a typo, say ₹80000 for two, from reaching the slide.

*The product team wants a badge on every card: "Popular" for 500 votes or more, "New" for the rest.* `np.where(condition, a, b)` is an if-else for every element:

```python
ratings = np.array([4.9, 4.1, 4.8])      # delivery partners
print(np.where(ratings >= 4.2, "Green Flag", "Red Flag"))
# ['Green Flag' 'Red Flag' 'Green Flag']
print(np.where(votes >= 500, "Popular", "New"))
```

Without the two values, `np.where(condition)` gives the **positions** (marketing wants the row numbers of the popular places, to send them an offer) where the condition is True, inside a tuple (one array per direction):

```python
print(np.where(votes >= 500))         # (array([0, 1, 2, 6, 8]),)
print(np.where(clean_costs > 1000))   # (array([], dtype=int64),)  no match, no error
```

In Colab, `help(np.where)` shows the documentation.

## 9. Sorting

*The product team wants a "Sort by: Popularity" button, and the manager asks which places are the most popular and what they cost.*

```python
print(np.sort(votes))          # [  88  166  286  324  402  504  775  787  918 2556]
print(np.sort(votes)[::-1])    # largest first: sort, then reverse with a step of -1
print(np.argsort(votes))       # [3 4 5 7 9 8 0 1 2 6]
```

`np.sort` gives a sorted copy; `votes` does not change. `np.sort(votes, reverse=True)` is a TypeError: `reverse=True` belongs to Python's `sorted()`.

`np.argsort` gives the **positions** in sorted order: position 3 holds the smallest vote, 88. Use those positions on a parallel array and every value stays with its restaurant:

```python
order = np.argsort(votes)[::-1]
print(votes[order][:3])         # [2556  918  787]
print(clean_costs[order][:3])   # [600. 800. 800.]
```

The top three cost ₹600, ₹800 and ₹800: the most popular place is not the costliest.

In 2D, `axis=1` sorts each row and `axis=0` sorts each column, each on its own. On a data table that is dangerous: `np.sort(data, axis=0)` gives a row of 166 votes and ₹500, a restaurant that does not exist (the 166-vote place costs ₹600). To sort whole rows, use argsort on one column: `data[np.argsort(data[:, 0])]`.

**Numbers stored as text sort like words.** `np.sort(np.array(["800", "1200", "300", "550"]))` gives `['1200' '300' '550' '800']`: text is compared letter by letter, so a "cost: low to high" list on text would show the ₹1200 place first. Check the dtype before you sort.

## Recap

- `np.arange(start, end, step)` makes numbers. `[start:end:step]` slices with a jump. A negative step reverses.
- **A mask** is an array of True and False. Inside the brackets it filters. Join two conditions with `&`, each in round brackets.
- `np.column_stack` makes a table: each array becomes a column. `reshape` refills the numbers; rows × columns must equal the size; `-1` works out one side. To swap rows and columns, use `.T`.
- `sum`, `mean`, `max`, `min`, `std` summarise. `axis=0`: one answer per column. `axis=1`: one per row.
- `np.any`, `np.all`: one True or False. `np.where(c, a, b)`: if-else for every element. `np.where(c)`: the positions.
- `np.sort` gives the values; `np.argsort` gives the positions, to keep parallel arrays together. Text sorts like words.

## The business questions, answered

Ten restaurants only: a sample, not the whole city.

| The question | Our answer |
|---|---|
| Are popular places more expensive? | Yes, here: ₹710 for two against ₹540. |
| Popular **and** affordable? | Two places: 2556 votes at ₹600, 504 votes at ₹550. |
| The weekly summary | 6806 votes in all, ₹625 average, 2556 the most votes, ₹300 the cheapest. |
| Can a restaurant promise 30 minutes? | A can (std 1.63). B cannot (std 16.33). |
| Riders for the weekend | Most on Saturday: 93 orders. |
| Which cards get "Popular"? | Five: positions 0, 1, 2, 6 and 8. |
| The most popular places, and their cost | 2556, 918 and 787 votes: ₹600, ₹800, ₹800. |

**Next class:** maths on whole arrays: matrix multiplication, broadcasting and `np.tile`.
