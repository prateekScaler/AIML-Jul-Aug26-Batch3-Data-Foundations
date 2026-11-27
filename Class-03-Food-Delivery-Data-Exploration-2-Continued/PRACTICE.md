# Class 3 · Practice

26 questions on today's tools, all on new data: a cloud kitchen's menu, its outlets, its riders. Each starts with the question someone in the business is asking. The bands go from warm-up to a **Challenge** band that joins several tools in one answer.

For each one: read the code, say the answer aloud, and only then open it. Every answer here was checked by running the code. You need only `import numpy as np` at the top.

Two data sets come back often. The kitchen's menu:

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
```

And the orders at its 4 outlets (rows) from Monday to Friday (columns):

```python
week = np.array([[42, 38, 45, 50, 61],
                 [30, 28, 33, 29, 40],
                 [55, 52, 58, 60, 72],
                 [18, 20, 17, 22, 25]])
```

## Warm-up: arange and steps

### 1. Table numbers

*A new restaurant has 12 tables and wants them numbered 1 to 12 on the floor plan.*

What does this print?

```python
print(np.arange(1, 13))
```

<details><summary>Answer</summary>

```text
[ 1  2  3  4  5  6  7  8  9 10 11 12]
```

Start at 1, stop **before** 13. To include 12, the end must be 13.

</details>

### 2. Delivery slots

*The kitchen takes orders for slots every 15 minutes, from minute 0 of the hour to minute 45.*

What does this print?

```python
print(np.arange(0, 60, 15))
```

<details><summary>Answer</summary>

```text
[ 0 15 30 45]
```

Jump by 15 and stop before 60: 0, 15, 30, 45. Minute 60 is the next hour, so leaving it out is right here.

</details>

### 3. Every third order

*The quality team checks every third order of the day, starting from the second one.*

What does this print?

```python
order_ids = np.arange(101, 116)
print(order_ids[1::3])
```

<details><summary>Answer</summary>

```text
[102 105 108 111 114]
```

Position 1 holds 102. Then every third position: 4, 7, 10, 13, which hold 105, 108, 111, 114.

</details>

### 4. Latest first

*The rider app lists today's 5 trips, oldest first. The rider wants the latest first.*

What does this print?

```python
trips = np.array([3.2, 5.0, 1.8, 4.4, 2.6])
print(trips[::-1])
```

<details><summary>Answer</summary>

```text
[2.6 4.4 1.8 5.  3.2]
```

A step of -1 starts at the last trip and walks left. `5.` is 5.0: one type only, so every value is a float.

</details>

## Core: masks

### 5. How many are cheap?

*The kitchen owner: how many dishes cost under ₹100?*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print((price < 100).sum())
```

<details><summary>Answer</summary>

```text
2
```

`price < 100` is True for Poha (60) and Paratha (90). True counts as 1, so the sum of the mask counts them: 2.

</details>

### 6. Best sellers

*The owner wants the names of the dishes with more than 100 orders this week.*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print(dishes[orders > 100])
```

<details><summary>Answer</summary>

```text
['Poha' 'Paratha' 'Dosa']
```

The mask comes from `orders` and filters `dishes`, position by position: positions 0, 2 and 4 pass.

</details>

### 7. Price of a best seller

*Are the best sellers cheap? Average price of the dishes with more than 100 orders, against the rest.*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print(price[orders > 100].mean())
print(price[orders <= 100].mean())
```

<details><summary>Answer</summary>

```text
86.66666666666667
173.33333333333334
```

Best sellers: (60 + 90 + 110) ÷ 3, about ₹86.7. The rest: (180 + 220 + 120) ÷ 3, about ₹173.3. Here, cheap dishes sell more: a hint for the menu, not proof.

</details>

### 8. A "Value combo"

*Marketing wants a "Value combo" banner: dishes under ₹150 with at least 100 orders.*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print(dishes[(price < 150) & (orders >= 100)])
```

<details><summary>Answer</summary>

```text
['Poha' 'Paratha' 'Dosa']
```

Both conditions must be True. Khichdi is under ₹150 but has only 40 orders, so it is out. Each condition needs its own round brackets.

</details>

### 9. and, not &

*A teammate writes the "Value combo" filter like this. What happens?*

What happens?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print(dishes[(price < 150) and (orders >= 100)])
```

<details><summary>Answer</summary>

```text
ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all()
```

`and` wants one True or False, but each side is an array of six. Use `&` to join masks element by element.

</details>

## Core: tables, reshape and axis

### 10. One sheet

*The owner wants price and orders side by side, one row per dish. What is the shape, and what is row 3?*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
menu = np.column_stack((price, orders))
print(menu.shape)
print(menu[3])
```

<details><summary>Answer</summary>

```text
(6, 2)
[220  60]
```

Each array becomes a column: 6 rows, 2 columns. Row 3 is Biryani: ₹220 and 60 orders. Both are integers here, so no `.` appears.

</details>

### 11. Busiest weekday

*Operations: total orders per weekday across the 4 outlets, to plan riders.*

What does this print?

```python
week = np.array([[42, 38, 45, 50, 61],
                 [30, 28, 33, 29, 40],
                 [55, 52, 58, 60, 72],
                 [18, 20, 17, 22, 25]])
print(week.sum(axis=0))
```

<details><summary>Answer</summary>

```text
[145 138 153 161 198]
```

`axis=0` adds down the rows (the outlets), leaving one total per weekday. Friday, 198, is the busiest.

</details>

### 12. Busiest outlet

*The manager: total orders per outlet over the 5 days.*

What does this print?

```python
week = np.array([[42, 38, 45, 50, 61],
                 [30, 28, 33, 29, 40],
                 [55, 52, 58, 60, 72],
                 [18, 20, 17, 22, 25]])
print(week.sum(axis=1))
```

<details><summary>Answer</summary>

```text
[236 160 297 102]
```

`axis=1` adds across each row, leaving one total per outlet. Outlet 2 is the busiest, outlet 3 the quietest.

</details>

### 13. Shapes after axis

*Before writing a report, check how many numbers each line gives.*

What does this print?

```python
week = np.array([[42, 38, 45, 50, 61],
                 [30, 28, 33, 29, 40],
                 [55, 52, 58, 60, 72],
                 [18, 20, 17, 22, 25]])
print(week.shape)
print(week.mean(axis=0).shape, week.mean(axis=1).shape)
```

<details><summary>Answer</summary>

```text
(4, 5)
(5,) (4,)
```

Shape `(4, 5)`: 4 outlets, 5 days. `axis=0` removes the 4 rows, leaving 5 numbers (one per day). `axis=1` removes the 5 columns, leaving 4 (one per outlet).

</details>

### 14. Cards in a row

*The app shows 12 dish cards. A phone fits 3 in a row. How many rows?*

What does this print?

```python
cards = np.arange(1, 13)
print(cards.reshape(-1, 3).shape)
print(cards.reshape(-1, 3)[-1])
```

<details><summary>Answer</summary>

```text
(4, 3)
[10 11 12]
```

12 cards, 3 per row: `-1` works out 4 rows. The last row holds cards 10, 11 and 12.

</details>

### 15. Cards that do not fit

*And 5 cards in a row, on a tablet?*

What happens?

```python
cards = np.arange(1, 13)
print(cards.reshape(-1, 5))
```

<details><summary>Answer</summary>

```text
ValueError: cannot reshape array of size 12 into shape (5)
```

12 ÷ 5 is not a whole number. `-1` works out one side; it cannot make the numbers fit.

</details>

## Stretch: any, all, where and sorting

### 16. Data check

*Before the weekly report: is any price zero or less (a mistake)? Is every dish ordered at least 30 times?*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print(np.any(price <= 0), np.all(orders >= 30))
```

<details><summary>Answer</summary>

```text
False True
```

No price is zero or less, so `any` is False. The smallest order count is 40, so `all` is True.

</details>

### 17. Price labels

*The menu shows "Budget" for dishes up to ₹100 and "Premium" above.*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print(np.where(price <= 100, "Budget", "Premium"))
```

<details><summary>Answer</summary>

```text
['Budget' 'Premium' 'Budget' 'Premium' 'Premium' 'Premium']
```

One label per dish. Dosa costs ₹110, just above ₹100, so it is "Premium".

</details>

### 18. Which rows?

*Marketing wants the positions of the dishes with fewer than 70 orders, to plan an offer.*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
print(np.where(orders < 70))
```

<details><summary>Answer</summary>

```text
(array([3, 5]),)
```

Without the two values, `np.where` gives the positions: 3 (Biryani, 60) and 5 (Khichdi, 40), inside a tuple.

</details>

### 19. Cheapest first

*The app's "Price: low to high" sort. Which dish comes first, and which last?*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
order = np.argsort(price)
print(order)
print(dishes[order][0], dishes[order][-1])
```

<details><summary>Answer</summary>

```text
[0 2 4 5 1 3]
Poha Biryani
```

`argsort` gives the positions in price order. Used on `dishes`, every name stays with its price.

</details>

### 20. Ratings stored as text

*Ratings arrived as text. The app sorts them "high to low" before converting. What is first?*

What does this print?

```python
r = np.array(["4.5", "10.0", "3.8"])
print(np.sort(r)[::-1])
```

<details><summary>Answer</summary>

```text
['4.5' '3.8' '10.0']
```

Text is compared letter by letter: "4" > "3" > "1". So "10.0", a data error on a 5-star scale, hides at the end. Convert with `astype(float)` first, and then `any` would catch the 10.

</details>

## Challenge: put the tools together

### 21. Top two best sellers

*The owner: the names and prices of the two dishes with the most orders, most first.*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
top = np.argsort(orders)[::-1][:2]
print(dishes[top])
print(price[top])
```

<details><summary>Answer</summary>

```text
['Dosa' 'Poha']
[110  60]
```

argsort orders the positions by orders; `[::-1]` puts the most first; `[:2]` keeps two. The same positions on `dishes` and `price` keep each dish together.

</details>

### 22. Which day?

*Operations wants the name of the busiest weekday, not just the number.*

What does this print?

```python
week = np.array([[42, 38, 45, 50, 61],
                 [30, 28, 33, 29, 40],
                 [55, 52, 58, 60, 72],
                 [18, 20, 17, 22, 25]])
days = np.array(["Mon", "Tue", "Wed", "Thu", "Fri"])
totals = week.sum(axis=0)
print(days[totals == totals.max()])
```

<details><summary>Answer</summary>

```text
['Fri']
```

Three tools: `axis=0` totals per day, `max` finds 198, and the mask `totals == totals.max()` picks its position in `days`.

</details>

### 23. Can we promise 40 minutes?

*Delivery times came as text like "35 min". Clean them, then decide: average, spread, and is every delivery within 40 minutes?*

What does this print?

```python
t = np.array(["35 min", "38 min", "41 min", "36 min", "40 min"])
mins = np.char.strip(t, " min").astype(int)
print(mins.mean(), mins.std().round(2))
print(np.all(mins <= 40))
```

<details><summary>Answer</summary>

```text
38.0 2.28
False
```

`np.char.strip(t, " min")` removes the characters space, m, i and n from both ends, leaving the digits. Average 38, small spread, but one delivery took 41: `all` is False. A "within 40 minutes" promise would have been broken once in five.

</details>

### 24. Four weeks in one line

*The owner exported 4 weeks of daily orders as one long row of 28 numbers. Turn it into one row per week and find each week's total.*

What does this print?

```python
daily = np.arange(10, 38)
weeks = daily.reshape(4, -1)
print(weeks.shape)
print(weeks.sum(axis=1))
```

<details><summary>Answer</summary>

```text
(4, 7)
[ 91 140 189 238]
```

28 numbers in 4 rows: `-1` works out 7 columns, one per day. `axis=1` gives one total per week. (The data is made up with `np.arange`, so orders grow by one a day.)

</details>

### 25. Rank the sheet

*Rank the menu sheet by orders, most first, keeping every row one dish. Which line is right? Print the top row of each.*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
menu = np.column_stack((price, orders))
print(np.sort(menu, axis=0)[::-1][0])
print(menu[np.argsort(menu[:, 1])[::-1]][0])
```

<details><summary>Answer</summary>

```text
[220 150]
[110 150]
```

The first line sorts each column on its own: ₹220 with 150 orders is no real dish. The second sorts whole rows by orders: Dosa, ₹110, 150 orders. Use argsort on one column to rank a table.

</details>

### 26. Count a segment

*Marketing: how many "Premium" dishes (over ₹100) still sold more than 50?*

What does this print?

```python
dishes = np.array(["Poha", "Thali", "Paratha", "Biryani", "Dosa", "Khichdi"])
price  = np.array([60, 180, 90, 220, 110, 120])
orders = np.array([140, 85, 120, 60, 150, 40])
premium = price > 100
print((premium & (orders > 50)).sum())
print(dishes[premium & (orders > 50)])
```

<details><summary>Answer</summary>

```text
3
['Thali' 'Biryani' 'Dosa']
```

Name the first mask, join it with `&`, and the sum counts the Trues. Khichdi is premium but sold only 40.

</details>

## Make your own

Ask the AI assistant in Colab: *"Give me a NumPy table of 5 shops by 7 days of sales, and 5 business questions on masks, axis, np.where and argsort. Do not show the answers."* Answer on paper, then run each one. If the assistant and NumPy disagree, NumPy is right.
