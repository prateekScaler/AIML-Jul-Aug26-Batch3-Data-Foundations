# Class 4 · Practice

17 questions on today's tools, all on new data: a bakery chain with four items and three outlets. Each starts with the question someone in the business is asking. The bands go from warm-up to a **Challenge** band that joins several tools in one answer.

For each one: read the code, say the answer aloud, and only then open it. Every answer here was checked by running the code. You need only `import numpy as np` at the top.

The bakery's items and prices, and yesterday's orders (one row per outlet, one column per item), come back often:

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
```

## Warm-up: element-wise maths

### 1. Line totals

*A customer buys 3 buns, 1 cake, 2 puffs and 5 cookies. What is each line of the bill?*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
qty = np.array([3, 1, 2, 5])
print(qty * price)
```

<details><summary>Answer</summary>

```text
[120 120 180 125]
```

Position by position: 3 × 40, 1 × 120, 2 × 90, 5 × 25.

</details>

### 2. A price rise

*Flour costs more: every price goes up 12%.*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
print(price * 1.12)
```

<details><summary>Answer</summary>

```text
[ 44.8 134.4 100.8  28. ]
```

One number, used for every element. The result is float because 1.12 is a float: one type only.

</details>

### 3. Stock left

*Morning stock and items sold so far. What is left on the shelf?*

What does this print?

```python
stock = np.array([50, 20, 30, 80])
sold = np.array([12, 20, 7, 33])
print(stock - sold)
```

<details><summary>Answer</summary>

```text
[38  0 23 47]
```

`-` pairs positions like `*`. Cakes are sold out: 0 left, time to bake more.

</details>

### 4. Two weights for four items

*A teammate multiplies the 4 prices by 2 weights.*

What happens?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
print(price * np.array([1, 2]))
```

<details><summary>Answer</summary>

```text
ValueError: operands could not be broadcast together with shapes (4,) (2,)
```

Four values and two have no position-by-position pairing. The error names both shapes.

</details>

## Core: matrix multiplication

### 5. One bill

*The same customer: 3 buns, 1 cake, 2 puffs, 5 cookies. The bill in one number.*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
qty = np.array([3, 1, 2, 5])
print(qty @ price)
```

<details><summary>Answer</summary>

```text
545
```

`@` multiplies in pairs and adds: 120 + 120 + 180 + 125 = 545.

</details>

### 6. Revenue per outlet

*Yesterday's orders: one row per outlet, one column per item. Revenue of each outlet?*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
print(orders @ price)
```

<details><summary>Answer</summary>

```text
[4480 3865 5020]
```

`(3, 4) @ (4,)` gives `(3,)`: one total per outlet. Outlet 0: 30 × 40 + 10 × 120 + 12 × 90 + 40 × 25 = 4480. Outlet 2 earns the most.

</details>

### 7. Predict the shape

*Two price lists, this month and next month, as two columns. What shape is the result?*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
two_lists = np.ones((4, 2))
print((orders @ two_lists).shape)
```

<details><summary>Answer</summary>

```text
(3, 2)
```

`(3, 4) @ (4, 2)`: the middle 4s match, the outside numbers stay: one revenue per outlet per month.

</details>

### 8. A price list that does not fit

*Someone uses a 3-item price list on the 4-item orders.*

What happens?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
print(orders @ np.array([40, 120, 90]))
```

<details><summary>Answer</summary>

```text
ValueError: matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc signature (n?,k),(k,m?)->(n?,m?) (size 3 is different from 4)
```

`(3, 4) @ (3,)`: 4 items in each order row, 3 prices. The middle numbers must match.

</details>

### 9. dot with one number

*Prices for a "buy 2" pack, written with `np.dot`.*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
print(np.dot(price, 2))
```

<details><summary>Answer</summary>

```text
[ 80 240 180  50]
```

`np.dot` with a single number multiplies every element. `price @ 2` would give an error. Clearer still: `price * 2`.

</details>

### 10. Row times column

*Check one product by hand before you trust the code.*

What does this print?

```python
A = np.array([[1, 0], [2, 1]])
B = np.array([[3, 4], [5, 6]])
print(A @ B)
```

<details><summary>Answer</summary>

```text
[[ 3  4]
 [11 14]]
```

Row 1 of A with column 0 of B: 2 × 3 + 1 × 5 = 11. With column 1: 2 × 4 + 1 × 6 = 14. `A * B` would give `[[ 3  0] [10  6]]`.

</details>

## Core: broadcasting

### 11. A fee per outlet

*Outlets 0 and 2 add a delivery charge per item ordered online: ₹5 and ₹10. Outlet 1 adds nothing.*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
fee = np.array([[5], [0], [10]])
print(orders + fee)
```

<details><summary>Answer</summary>

```text
[[35 15 17 45]
 [18  6 20 25]
 [35 25 18 70]]
```

`(3, 4) + (3, 1)`: from the right, 4 against 1, so the column stretches across. Each outlet's number is added to its whole row. (The table here holds the numbers, not rupees: it is the shape that matters.)

</details>

### 12. A combo preview

*Preview every item at full price, 10% off and 25% off.*

What does this print?

```python
p = np.array([[40], [120], [90], [25]])
print(p * np.array([[1.0, 0.9, 0.75]]))
```

<details><summary>Answer</summary>

```text
[[ 40.    36.    30.  ]
 [120.   108.    90.  ]
 [ 90.    81.    67.5 ]
 [ 25.    22.5   18.75]]
```

`(4, 1) * (1, 3)` gives `(4, 3)`: row = item, column = price level. A cookie at 25% off is ₹18.75.

</details>

### 13. One number per outlet, wrong shape

*Someone writes the outlet fees as a plain 1D array.*

What happens?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
print(orders + np.array([5, 0, 10]))
```

<details><summary>Answer</summary>

```text
ValueError: operands could not be broadcast together with shapes (3,4) (3,)
```

From the right: 4 against 3. Not equal, neither is 1. One number per row needs shape `(3, 1)`.

</details>

## Challenge: put the tools together

### 14. The cheapest supplier

*Three suppliers quote prices for the 4 items (one column each). For the order of 3, 1, 2, 5 items, which supplier is cheapest?*

What does this print?

```python
qty = np.array([3, 1, 2, 5])
quotes = np.array([[ 40,  38,  42],
                   [120, 125, 110],
                   [ 90,  85,  95],
                   [ 25,  24,  26]])
total = qty @ quotes
print(total)
print(np.where(total == total.min()))
```

<details><summary>Answer</summary>

```text
[545 529 556]
(array([1]),)
```

`(4,) @ (4, 3)` gives one total per supplier. `np.where` with one argument gives the position of the smallest: supplier 1, ₹529.

</details>

### 15. Rank the menu

*A "Bestsellers" shelf: 60% rating, 40% orders (in hundreds, all outlets). Rank the items, best first.*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
rating = np.array([4.2, 4.6, 3.9, 4.4])
feat = np.array([rating, orders.sum(axis=0) / 100]).T
score = feat @ np.array([0.6, 0.4])
print(score.round(2))
print(items[np.argsort(score)[::-1]])
```

<details><summary>Answer</summary>

```text
[2.81 2.88 2.5  3.14]
['Cookie' 'Cake' 'Bun' 'Puff']
```

Column totals with `axis=0`, a `(4, 2)` feature table, `@` with weights, then argsort with `[::-1]`. Cookies win on volume (125 sold) even with a 4.4 rating.

</details>

### 16. Compare outlets fairly

*Each item's orders as a share of its best outlet: 1.0 marks the outlet that sold the most of it.*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
print((orders / orders.max(axis=0)).round(2))
```

<details><summary>Answer</summary>

```text
[[1.   0.67 0.6  0.67]
 [0.6  0.4  1.   0.42]
 [0.83 1.   0.4  1.  ]]
```

`orders.max(axis=0)` is `(4,)`, one maximum per item. Broadcasting divides every row by it. Outlet 1 sells the most puffs; outlet 2 leads in cakes and cookies.

</details>

### 17. Two ways, one answer

*Is revenue per outlet the same with `*` then a row sum, and with `@`?*

What does this print?

```python
items = np.array(["Bun", "Cake", "Puff", "Cookie"])
price = np.array([40, 120, 90, 25])
orders = np.array([[30, 10, 12, 40],
                   [18,  6, 20, 25],
                   [25, 15,  8, 60]])
way1 = (orders * price).sum(axis=1)
way2 = orders @ price
print(way1, np.array_equal(way1, way2))
```

<details><summary>Answer</summary>

```text
[4480 3865 5020] True
```

`orders * price` broadcasts the price row down: each line total. `.sum(axis=1)` adds each row. That is exactly what `@` does in one step.

</details>

## Make your own

Ask the AI assistant in Colab: *"Give me a small NumPy table of 3 shops by 4 products and 5 business questions on @ and broadcasting. Do not show the answers."* Answer on paper, then run each one. If the assistant and NumPy disagree, NumPy is right.
