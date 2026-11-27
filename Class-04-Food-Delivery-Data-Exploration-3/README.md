# Class 4 · Food Delivery Data Exploration and Analysis 3

The fourth class of Data Foundations: maths on whole arrays. Element-wise operations, matrix multiplication with `@` and broadcasting, all on the same 10 restaurants, now with a rating for each.

We are the data team of a food delivery app. Every tool today answers a question the business asks:

| Who asks | The question | The tool |
|---|---|---|
| The manager (from last class) | Which places are most popular, and what do they cost? | `np.sort`, `np.argsort` (a short carry-over) |
| The manager | A 5.0 from 2 people or a 4.8 from 1000: which do we trust? | `*`, element by element |
| Restaurants | Every price goes up 5%. The new prices? | array × number |
| The app | What does each customer's cart cost, at each restaurant? | matrix multiplication, `@` |
| The ranking team | One score per restaurant from rating and votes | `@` with weights |
| Pricing | Add fees to every price; preview a 10, 20, 50% sale | broadcasting |

The answers are collected at the end, in [The business questions, answered](#the-business-questions-answered).

**By the end of this class you can:**

- Sort with `np.sort`, and keep parallel arrays together with `np.argsort` (carried over from Class 3).
- Do maths on whole arrays with `*`, `+`, `-`, `/`, with another array or with one number.
- Multiply matrices with `@`, `np.dot` and `np.matmul`, and predict the shape of the result.
- Say when broadcasting works (shapes compared from the right: equal, or 1) and when it fails.

## How to use this folder

| File | What it is |
|---|---|
| [`index.html`](index.html) | The interactive class page. **Step by step** shows a little at a time (press → or Next). **Show everything** gives the whole class on one page. |
| [`code/`](code/) | Every example from the class as a runnable file, in order. |
| [`PRACTICE.md`](PRACTICE.md) | 17 practice questions on new business data, from warm-up to a Challenge band, each with an explained answer. |
| [`Data-Foundations-Class-04-Food-Delivery-Data-Exploration-3.ipynb`](Data-Foundations-Class-04-Food-Delivery-Data-Exploration-3.ipynb) | The same examples as one notebook. Upload it to [Google Colab](https://colab.research.google.com/) and run the cells, nothing to install. |

Run the code in Google Colab, or on your own machine after `pip install numpy scikit-learn matplotlib`.

---

## 0. Carry-over from Class 3: sorting

Class 3 did not reach sorting, so this class starts with it.

*The product team wants a "Sort by: Popularity" button, most votes first.*

```python
votes = np.array([775, 787, 918, 88, 166, 286, 2556, 324, 504, 402])
costs = np.array([800., 800., 800., 300., 600., 600., 600., 700., 550., 500.])

print(np.sort(votes))          # [  88  166  286  324  402  504  775  787  918 2556]
print(np.sort(votes)[::-1])    # largest first: sort, then reverse with a step of -1
```

`np.sort` gives a sorted copy; `votes` does not change. `np.sort(votes, reverse=True)` is a TypeError: `reverse=True` belongs to Python's `sorted()`. NumPy 2.5 added `np.sort(votes, descending=True)` (and the same for `np.argsort`, [NumPy docs](https://numpy.org/doc/2.5/reference/generated/numpy.sort.html)), but Colab runs NumPy 2.0.2 ([Colab runtime list](https://research.google.com/colaboratory/runtime-version-faq.html)), where it is a TypeError too. In Colab, use `[::-1]`.

*The manager, since Class 1: which places are the most popular, and what do they cost?* `np.argsort` gives the **positions** in sorted order. Use them on the parallel array and every cost stays with its restaurant:

```python
print(np.argsort(votes))        # [3 4 5 7 9 8 0 1 2 6]
order = np.argsort(votes)[::-1]
print(votes[order][:3])         # [2556  918  787]
print(costs[order][:3])         # [600. 800. 800.]
```

The top three cost ₹600, ₹800 and ₹800: the most popular place is not the costliest.

**Sorting a table.** On the manager's sheet `data = np.column_stack((votes, costs))`, `np.sort(data, axis=0)` sorts each column on its own and gives a row of 166 votes and ₹500, a restaurant that does not exist (the 166-vote place costs ₹600). To rank whole rows, use argsort on one column, in three steps:

```python
col = data[:, 0]          # 1. the votes column
order = np.argsort(col)   # 2. row numbers, fewest votes first
ranked = data[order]      # 3. whole rows, in that order
print(order)              # [3 4 5 7 9 8 0 1 2 6]
print(ranked[:3])         # [[ 88. 300.] [166. 600.] [286. 600.]]
```

Each row is still one restaurant. In one line: `data[np.argsort(data[:, 0])]`; add `[::-1]` after the argsort for most votes first.

**Numbers stored as text sort like words.** `np.sort(np.array(["800", "1200", "300", "550"]))` gives `['1200' '300' '550' '800']`: a "cost: low to high" list on text would show the ₹1200 place first. Check the dtype, convert, then sort.

## Our data today

The same 10 restaurants, now with a rating out of 5. The costs are already clean floats. These are practice values for the class.

```python
votes = np.array([775, 787, 918, 88, 166,
                  286, 2556, 324, 504, 402])
ratings = np.array([4.1, 4.1, 3.8, 3.7, 3.8,
                    3.8, 4.6, 4.0, 4.2, 4.1])
costs = np.array([800., 800., 800., 300., 600.,
                  600., 600., 700., 550., 500.])
```

Three parallel arrays: position 6 is the same restaurant in all three (4.6 stars, 2556 votes, ₹600).

## 1. Element-wise maths

*The manager: a 5.0 from 2 people, or a 4.8 from 1000 people. Which rating do we trust more?* The 4.8, because a thousand people agree. A score that grows with both rating and votes says so:

```python
trust = votes * ratings
print(trust)
# [ 3177.5  3226.7  3488.4   325.6   630.8  1086.8 11757.6  1296.   2116.8
#   1648.2]
```

`*` between two arrays of the same shape pairs up the same positions: 775 × 4.1 = 3177.5, and so on. This is **element-wise** multiplication. `+`, `-` and `/` work the same way. See it step by step in the [Element-wise visualizer](https://aiml-v-izs.vercel.app/v/fd-r8w5n3p2/elementwise?embed=1) (pick multiplication).

**A catch.** `np.argsort(trust)[::-1]` and `np.argsort(votes)[::-1]` give the same order, `[6 2 1 0 8 9 7 5 4 3]`. Votes are in hundreds and thousands, ratings are about 4, so the votes decide everything and the rating changes nothing. A fair score needs the numbers on one scale, and weights: see the ranking score in section 2.

**An array and one number.** *Restaurants: every price goes up 5%.* The number is used for every element:

```python
print(costs * 1.05)
# [840.  840.  840.  315.  630.  630.  630.  735.  577.5 525. ]
```

One line updates 10 prices or 80,000, and no price is forgotten.

**Shapes that do not fit.** `votes * np.array([1, 2, 3, 4])` gives `ValueError: operands could not be broadcast together with shapes (10,) (4,)`. Ten values and four have no position-by-position pairing, and NumPy refuses rather than guess. Read the error: it names both shapes.

| You write | Result |
|---|---|
| array * number | works: the number is used for every element |
| array * array, same shape | works: position by position |
| array * array, shapes that do not fit | an error (some do fit: see broadcasting) |

## 2. Matrix multiplication: `@`

*The app: a customer orders 2 biryanis, 1 naan and 3 lassis at ₹250, ₹40 and ₹60. What is the bill?* A row (the cart) times a column (the prices): multiply in pairs, then add.

```python
prices = np.array([250, 40, 60])
cart = np.array([2, 1, 3])
print(cart * prices)     # [500  40 180]  each line of the bill
print(cart @ prices)     # 720            500 + 40 + 180
```

`@` is **matrix multiplication**. With one row per customer, one line bills everyone:

```python
carts = np.array([[2, 1, 3],
                  [1, 4, 0],
                  [0, 2, 2]])
print(carts @ prices)    # [720 410 200]
```

*The same carts at two restaurants, A and B: where is each cart cheaper?* Put one price list per column:

```python
price_lists = np.array([[250, 220],
                        [ 40,  50],
                        [ 60,  70]])
bills = carts @ price_lists
print(bills)
# [[720 700]
#  [410 420]
#  [200 240]]
```

Row = customer, column = restaurant. Cell (0, 1) is row 0 of `carts` times column 1 of `price_lists`: 2 × 220 + 1 × 50 + 3 × 70 = 700. Customer 0 pays less at B; customers 1 and 2 pay less at A.

**See it cell by cell.** The [class page](https://prateekscaler.github.io/AIML-Jul-Aug26-Batch3-Data-Foundations/Class-04-Food-Delivery-Data-Exploration-3/) has a matrix multiplication visualizer on the "Two restaurants" step: press *Next cell* (or click a cell of the result) and it highlights the row of `carts`, the column of `price_lists`, and builds the sum, for this example and for a 2 × 2 one. The content team's [Matrix multiply visualizer](https://aiml-v-izs.vercel.app/v/fd-r8w5n3p2/matmul?embed=1) shows the same idea.

**The shape rule.** `(A, B) @ (B, C)` gives `(A, C)`: the middle numbers must match, the outside numbers stay.

| Shapes | Result |
|---|---|
| `(3, 3) @ (3, 2)` | `(3, 2)` |
| `(3, 4) @ (4, 3)` | `(3, 3)` |
| `(10, 2) @ (2,)` | `(10,)` |
| `(3, 3) @ (4,)` | an error: 3 ≠ 4 |

A 3-dish cart against a 4-price list gives `ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0 … (size 4 is different from 3)`. A dish without a price is a billing bug; the shape check stops it.

**`@`, `np.dot` and `np.matmul`.** On matrices all three give the same answer (`@` is short for `np.matmul`). One difference: with a single number, `a @ 5` and `np.matmul(a, 5)` give an error, while `np.dot(a, 5)` multiplies every element (`[ 5 10 15 20]` for `a = np.array([1, 2, 3, 4])`). For an array times a number, just write `a * 5`.

| | `*` | `@` or `np.matmul` | `np.dot` |
|---|---|---|---|
| array × number | every element | an error | every element |
| two arrays | position by position | row times column, then add | row times column, then add |

**`*` is not `@`.** For `A = [[1, 2], [3, 4]]` and `B = [[5, 6], [7, 8]]`, `A * B` is `[[ 5 12] [21 32]]` (position by position) and `A @ B` is `[[19 22] [43 50]]` (1 × 5 + 2 × 7 = 19, and so on).

**A ranking score is a matrix multiplication.** *The ranking team: one score per restaurant, 70% rating and 30% popularity (votes in thousands).*

```python
features = np.array([ratings, votes / 1000]).T   # (10, 2)
weights = np.array([0.7, 0.3])                   # (2,)
score = features @ weights                       # (10,)
print(score.round(2))
# [3.1  3.11 2.94 2.62 2.71 2.75 3.99 2.9  3.09 2.99]
```

Row 0: 0.7 × 4.1 + 0.3 × 0.775 = 3.10. Position 6 ranks first. Change the weights and the ranking changes: the weights are a business decision written as numbers.

**Why divide the votes by 1000?** Without it, `np.array([ratings, votes]).T @ weights` ranks the restaurants in the plain vote order again, `[6 2 1 0 8 9 7 5 4 3]`: votes in the hundreds drown a rating of 4. With it the order is `[6 1 0 8 9 2 7 5 4 3]`, and position 2 (918 votes, but 3.8 stars) drops from 2nd to 6th. Features must be on a similar scale before you weigh them; real ranking and machine learning systems always scale first. The same operation, at a much larger size, is a layer of a neural network: the input times a weight matrix.

## 3. Broadcasting

**Broadcasting** is how NumPy does maths on two arrays of different shapes: it stretches the smaller one to fit the bigger one. **Why:** one fee, tax or discount can go onto every price in one line, with no loop and no copies of the data.

**Start in 1D.** *Three dishes cost ₹100, ₹200 and ₹300; add a ₹20 packing fee to each.*

```python
prices = np.array([100, 200, 300])
print(prices + 20)                       # [120 220 320]  the 20 stretches to 3
print(prices + np.array([10, 20, 30]))   # [110 220 330]  same length: no stretch
print(prices + np.array([10, 20]))
# ValueError: operands could not be broadcast together with shapes (3,) (2,)
```

| Second array | Length | Works? |
|---|---|---|
| `20` | one number | yes: stretches to 3 |
| `np.array([20])` | 1 | yes: a 1 stretches |
| `np.array([10, 20, 30])` | 3 | yes: same length |
| `np.array([10, 20])` | 2 | no: the third price has no partner, and 2 is not 1 |

**Then tables.**

*Pricing: add the ₹30 delivery fee to every bill.* `bills` is `(3, 2)` and 30 is one number:

```python
print(bills + 30)
# [[750 730]
#  [440 450]
#  [230 270]]
```

NumPy acts as if 30 were a `(3, 2)` table of 30s. This is **broadcasting**: the smaller array is stretched, like a rubber band, only for the operation. No table of 30s is stored. See it in the [Broadcasting visualizer](https://aiml-v-izs.vercel.app/v/fd-z1y6t8h5/broadcasting?embed=1).

**A row stretches down.** *Packaging costs ₹10, ₹5 and ₹15 per dish; add it to both outlets' menus.*

```python
menu = np.array([[250, 40, 60],
                 [230, 45, 65]])
packing = np.array([10, 5, 15])
print(menu + packing)
# [[260  45  75]
#  [240  50  80]]
```

**A column and a row make a table.** *A festival sale preview: three dishes at ₹100, ₹200 and ₹400, each at 10%, 20% and 50% off.*

```python
prices = np.array([[100], [200], [400]])   # (3, 1)
off = np.array([[0.9, 0.8, 0.5]])          # (1, 3)
print(prices * off)
# [[ 90.  80.  50.]
#  [180. 160. 100.]
#  [360. 320. 200.]]
```

Both stretch: the column across, the row down. Row = dish, column = discount. **How to know what stretches: look for the 1.** `prices` is `(3, 1)`: 1 column, so it stretches right to 3 columns. `off` is `(1, 3)`: 1 row, so it stretches down to 3 rows. A side that is not 1 never stretches.

**When it works, and when it fails.** Line the two shapes up at the right end and compare one number at a time. Equal, or one of them is 1: it fits, and the 1 stretches. Anything else: an error.

| Shapes | Result |
|---|---|
| `(2, 3)` and `(2, 1)` | works: 3 vs 1 stretches, 2 vs 2 equal → `(2, 3)` |
| `(2, 3)` and `(3,)` | works: the row repeats → `(2, 3)` |
| `(3, 1)` and `(1, 3)` | works: both stretch → `(3, 3)` |
| `(2, 3)` and `(1, 2)` | error: 3 vs 2 |
| `(3,)` and `(2, 2)` | error: 3 vs 2 |

`np.array([[1, 2, 3], [4, 5, 6]]) + np.array([[10, 20]])` gives `ValueError: operands could not be broadcast together with shapes (2,3) (1,2)`: a packaging list for 2 dishes on a 3-dish menu fails loudly. To add one number per row of a `(4, 3)` table, give it shape `(4, 1)`, not `(4,)`.

## Recap

- `*`, `+`, `-`, `/` work element by element. An array and a number: the number goes to every element.
- `@` (or `np.matmul`, `np.dot`) multiplies a row by a column and adds. `(A, B) @ (B, C)` gives `(A, C)`. Only `np.dot` accepts a single number.
- **Broadcasting** stretches a smaller array when the shapes fit from the right: equal, or 1. Nothing is copied.
- Carried over: `np.sort` for values, `np.argsort` for positions; sort a table by whole rows.

## The business questions, answered

Our 10 practice restaurants.

| The question | Our answer |
|---|---|
| The most popular places, and their cost | `np.argsort(votes)[::-1]`: 2556, 918 and 787 votes, at ₹600, ₹800, ₹800. |
| Which rating do we trust? | The one with more people behind it: `votes * ratings`. Position 6 tops it, 11757.6. |
| Prices up 5% | `costs * 1.05`: ₹800 becomes ₹840. |
| Each cart, at each restaurant | `carts @ price_lists`: customer 0 is cheaper at B (₹700), customers 1 and 2 at A. |
| One ranking score | `features @ weights`: position 6 first, 3.99. |
| Fees and a sale preview | `bills + 30`; `prices * off` gives every dish at every discount. |

**Next class:** `np.tile` (real copies) and `np.vectorize` (your own rule on every element), which this class did not reach; then splitting and stacking arrays, and Pandas.
