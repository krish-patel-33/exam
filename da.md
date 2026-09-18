# Data Cleaning + Power BI Dashboard — Full Concept Guide

This guide explains **every concept** used in your practical exam sets (Hospital, Banking, Food Delivery, Hotel Booking) in detail, with working code you can adapt to any dataset by swapping column names. Each exam has two parts:

- **Q1 — Data Cleaning/Preprocessing & Validation** (NumPy + Pandas + Matplotlib)
- **Q2 — Power BI Dashboard** (DAX measures + visuals + slicers)

I'll use the **Hospital dataset (Set-1)** as the running example, but every technique here applies directly to Sets 2, 3, and 4 — only the column names change.

---

## Part A — Data Cleaning & Validation (Python)

### Step 0: Setup and loading data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("hospital_data.csv")

# Always inspect first — never clean blind
print(df.shape)          # (rows, columns)
print(df.info())         # data types + non-null counts
print(df.head())
print(df.describe(include="all"))   # stats for numeric + categorical
```

**Why:** `df.info()` tells you which columns have missing values (Non-Null Count < total rows) and whether a numeric column is wrongly stored as text (`object` dtype). `describe()` shows min/max/mean, which is your first clue for abnormal (outlier) values.

---

### Step 1: Handling missing values

Missing values appear as `NaN`, empty strings, or placeholder text like `"?"`, `"N/A"`, `"unknown"`.

```python
# 1. Find out how many missing values per column
print(df.isnull().sum())

# 2. Standardize hidden missing values first
df.replace(["?", "N/A", "n/a", "unknown", "", " "], np.nan, inplace=True)
```

Now choose a strategy **per column type**:

| Column type | Example | Strategy |
|---|---|---|
| Numeric, roughly normal | `Age` | Fill with **mean** or **median** (median if skewed/has outliers) |
| Numeric, skewed | `Total_Bill` | Fill with **median** |
| Categorical | `Gender`, `City` | Fill with **mode** (most frequent value) |
| ID column | `Patient_ID` | **Never fill** — if the ID itself is missing, drop the row (you can't fabricate an identity) |
| Too many missing (>50%) | any | Consider dropping the column entirely |

```python
# Numeric — median is safer than mean when outliers exist
df["Age"] = df["Age"].fillna(df["Age"].median())

# Categorical — mode
df["Payment_Mode"] = df["Payment_Mode"].fillna(df["Payment_Mode"].mode()[0])

# ID column — drop rows where the identifier itself is missing
df = df.dropna(subset=["Patient_ID"])
```

**Why median over mean?** Mean is pulled toward extreme values (e.g., one patient billed ₹50,00,000 skews the average). Median is the middle value and ignores outlier influence — safer as a default for billing/age fields.

---

### Step 2: Handling duplicate values

```python
# Full-row duplicates
print(df.duplicated().sum())
df = df.drop_duplicates()

# Duplicates on a key column only (e.g. same Patient_ID appearing twice
# with different other values — usually a data-entry error)
df = df.drop_duplicates(subset=["Patient_ID"], keep="first")
```

**Why `keep="first"`?** It's a convention — keep the earliest record. In a real system you'd instead keep the most *recently updated* row (sort by a timestamp column first if one exists), but for exam purposes, first/last is acceptable as long as you state your reasoning.

---

### Step 3: Correcting inconsistent categorical values

Real-world data almost never has clean, uniform text. `"Male"`, `"male "`, `"M"`, `"MALE"` are all the same value written four different ways.

```python
# See all unique values to spot inconsistency
print(df["Gender"].unique())
print(df["City"].unique())

# Standardize casing and whitespace
df["Gender"] = df["Gender"].str.strip().str.title()   # " male " -> "Male"

# Map known variants to one canonical value
gender_map = {"M": "Male", "MALE": "Male", "F": "Female", "FEMALE": "Female"}
df["Gender"] = df["Gender"].replace(gender_map)

# Same pattern for City, Visit_Type, Admission_Status
df["City"] = df["City"].str.strip().str.title()
df["Visit_Type"] = df["Visit_Type"].str.strip().str.title()
df["Admission_Status"] = df["Admission_Status"].str.strip().str.title()

# Verify — should now show one entry per real category
print(df["Gender"].unique())
```

**Why `.str.title()` before mapping?** It collapses casing differences (`"male"`, `"MALE"`, `"Male"` → `"Male"`) so your mapping dictionary needs far fewer entries.

---

### Step 4: Identifying and handling abnormal values (outliers)

"Abnormal" means values that are technically valid numbers but don't make real-world sense (e.g., `Age = 250`, `Length_of_Stay_Days = -3`).

**Method 1 — Domain logic (fastest, always do this first):**

```python
# Age must be between 0 and 120
df = df[(df["Age"] >= 0) & (df["Age"] <= 120)]

# Length of stay can't be negative
df = df[df["Length_of_Stay_Days"] >= 0]

# Satisfaction score usually on a fixed scale, e.g. 1–5
df = df[df["Satisfaction_Score"].between(1, 5)]
```

**Method 2 — IQR (Interquartile Range) rule — use when there's no obvious domain limit:**

```python
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return data[(data[column] >= lower) & (data[column] <= upper)]

df = remove_outliers_iqr(df, "Length_of_Stay_Days")
```

**How IQR works:** Q1 and Q3 are the 25th and 75th percentiles. IQR = Q3 − Q1 is the "typical spread" of the middle 50% of data. Anything more than 1.5×IQR beyond that range is statistically considered an outlier. This is the standard rule used in box plots.

> Prefer **capping/clipping** over deleting rows when you want to preserve dataset size:
> ```python
> df["Age"] = df["Age"].clip(lower=0, upper=100)
> ```

---

### Step 5: Validating and correcting billing/financial data

This is a **business-rule check** — numbers should relate to each other in a specific way.

```python
# Expected relationship: Net_Amount = Total_Bill - Insurance_Coverage
# Check where this doesn't hold
mismatch = df[abs((df["Total_Bill"] - df["Insurance_Coverage"]) - df["Net_Amount"]) > 1]
print(f"Rows with billing mismatch: {len(mismatch)}")

# Recalculate to fix instead of guessing/dropping — the two source
# columns (Total_Bill, Insurance_Coverage) are more trustworthy
# than a derived column (Net_Amount)
df["Net_Amount"] = df["Total_Bill"] - df["Insurance_Coverage"]

# Sanity checks — no negative money values
df = df[(df["Room_Charges"] >= 0) & (df["Total_Bill"] >= 0)]

# Insurance coverage can't exceed the total bill
df["Insurance_Coverage"] = df[["Insurance_Coverage", "Total_Bill"]].min(axis=1)
```

**Why recompute instead of drop?** If two of three related columns are reliable, it's better to derive the third than to lose the whole record. This is the same pattern used for `Final_Amount = Food_Amount + Delivery_Fee - Discount` (Set-3) or `Final_Amount = Gross_Amount - Discount` (Set-4) — identify the formula, verify it, recompute the derived column.

---

### Step 6: Date/Time handling (Set-3 style — Order_Date, Order_Time)

```python
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Order_Time"] = pd.to_datetime(df["Order_Time"], format="%H:%M:%S", errors="coerce").dt.time

# errors="coerce" turns unparseable dates into NaT (Not a Time)
# instead of crashing — then handle them like any other missing value
print(df["Order_Date"].isnull().sum())
df = df.dropna(subset=["Order_Date"])

# Feature engineering often expected here too:
df["Order_Year"] = df["Order_Date"].dt.year
df["Order_Month"] = df["Order_Date"].dt.month
df["Order_Weekday"] = df["Order_Date"].dt.day_name()
```

---

### Step 7: Validate with Matplotlib

The point of these charts is to **visually confirm the cleaning worked** — not just to make pretty pictures.

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Histogram — check Age distribution looks reasonable now (no 250s, no negatives)
axes[0, 0].hist(df["Age"], bins=20, color="skyblue", edgecolor="black")
axes[0, 0].set_title("Age Distribution")

# 2. Boxplot — visually confirm outliers are gone from Length_of_Stay_Days
axes[0, 1].boxplot(df["Length_of_Stay_Days"])
axes[0, 1].set_title("Length of Stay (Boxplot)")

# 3. Bar chart — confirm categorical values are now standardized (few, clean bars)
df["Gender"].value_counts().plot(kind="bar", ax=axes[1, 0], color="salmon")
axes[1, 0].set_title("Gender Count (post-cleaning)")

# 4. Scatter plot — sanity-check a relationship, e.g. Total_Bill vs Net_Amount
axes[1, 1].scatter(df["Total_Bill"], df["Net_Amount"], alpha=0.5)
axes[1, 1].set_title("Total Bill vs Net Amount")

plt.tight_layout()
plt.savefig("validation_charts.png")
plt.show()
```

**Chart-to-purpose cheat sheet** (use this logic for any of the four sets):

| Chart | Best for |
|---|---|
| Histogram | Distribution of one numeric column (Age, Delivery_Time_Min) |
| Boxplot | Spotting/confirming outliers are handled |
| Bar chart | Counts across categories (City, Room_Type, Cuisine) |
| Scatter plot | Relationship between two numeric columns (Room_Rate vs Final_Amount) |
| Line chart | Trend over time (Order_Date vs daily orders) |

---

### Step 8: Export cleaned data

```python
df.to_csv("hospital_data_cleaned.csv", index=False)
```

`index=False` is important — otherwise Pandas writes an extra unnamed column of row numbers into the CSV, which Power BI will treat as a real (junk) column.

---

## Part B — Power BI Dashboard

### Step 1: Load the cleaned CSV

1. Open Power BI Desktop → **Home → Get Data → Text/CSV**.
2. Select your cleaned CSV → **Load** (or **Transform Data** first if you want to double check types in Power Query).
3. In **Power Query Editor**, confirm each column's data type icon is correct (123 = number, calendar = date, ABC = text). Fix any that Power BI guessed wrong — this avoids DAX errors later.

### Step 2: Understand DAX measures (this is the concept students usually find hardest)

A **measure** is a calculation computed on the fly, based on whatever filters/slicers are currently applied — unlike a calculated column, which is computed once per row and stored.

Create measures via **Modeling → New Measure** (or right-click the table in the Fields pane).

```DAX
Total Patients = DISTINCTCOUNT(hospital_data_cleaned[Patient_ID])

Total Visits = COUNTROWS(hospital_data_cleaned)

Total Treatment Cost = SUM(hospital_data_cleaned[Room_Charges])

Total Bill Amount = SUM(hospital_data_cleaned[Total_Bill])

Average Satisfaction Score = AVERAGE(hospital_data_cleaned[Satisfaction_Score])
```

**Why `DISTINCTCOUNT` for Total Patients but `COUNTROWS` for Total Visits?** One patient can have multiple visit rows. `COUNTROWS` counts every row (= every visit). `DISTINCTCOUNT` counts unique `Patient_ID` values (= number of actual people). Mixing these up is the #1 mistake in this kind of exam — always ask "is this row-level or entity-level?"

Equivalent measures for the other three sets (same logic, different columns):

```DAX
-- Set-2 (Banking)
Total Customers      = DISTINCTCOUNT(banking_data[Customer_ID])
Total Loans           = COUNTROWS(banking_data)
Total Loan Amount     = SUM(banking_data[Loan_Amount])
Total Transaction Amount = SUM(banking_data[Transaction_Amount])
Average Credit Score  = AVERAGE(banking_data[Credit_Score])

-- Set-3 (Food Delivery)
Total Orders          = DISTINCTCOUNT(food_data[Order_ID])
Total Food Amount      = SUM(food_data[Food_Amount])
Total Discount         = SUM(food_data[Discount])
Average Delivery Time  = AVERAGE(food_data[Delivery_Time_Min])
Average Customer Rating = AVERAGE(food_data[Customer_Rating])

-- Set-4 (Hotel)
Total Bookings         = DISTINCTCOUNT(hotel_data[Booking_ID])
Total Guests            = SUM(hotel_data[Guests_Count])
Total Revenue           = SUM(hotel_data[Final_Amount])
Average Stay Duration   = AVERAGE(hotel_data[Occupancy_Nights])
Average Guest Rating    = AVERAGE(hotel_data[Guest_Rating])
```

### Step 3: Card visuals for the KPIs

1. **Insert → Visualizations → Card** (single-value icon).
2. Drag one measure (e.g. `Total Patients`) into the card's **Fields** well.
3. Repeat for each KPI, or use a **Multi-row card** to show several at once in one visual.
4. Format: **Format pane → Callout value** to control font size, and **Category label** for the label text underneath.

### Step 4: Choosing the right visual for each analysis point

This is a recurring rubric across all four exam sets ("create suitable visuals to analyse X, Y, Z") — the grading is about *matching the chart type to the data*, not just making any chart.

| What you're analysing | Best visual | Why |
|---|---|---|
| One categorical column's totals (Department, City, Cuisine) | **Bar/Column chart** | Easy comparison of discrete categories |
| Category as % of whole (Payment_Status, Loan_Status) | **Pie / Donut chart** | Shows proportion — use only for ≤5–6 categories |
| Trend over time (Booking_Date, Order_Date) | **Line chart** | Shows direction/trend clearly |
| Two numeric variables' relationship (Room_Rate vs Final_Amount) | **Scatter chart** | Shows correlation/clusters |
| Category broken down by sub-category (Department × Payment_Status) | **Stacked bar/column** | Shows composition within categories |
| Geographic comparison (City) | **Map / Filled map visual** | Instantly geographic |
| Table-level detail check | **Table/Matrix visual** | Lets you verify exact numbers |

Example — building one: **Insert → Column chart**, drag `Department` to **X-axis**, drag `Treatment_Cost` (set to Sum) to **Y-axis**. Add `Payment_Status` to **Legend** for a stacked breakdown.

### Step 5: Slicers for interactivity

1. **Insert → Slicer** visual.
2. Drag a filterable field into it — typically `City`, `Department`, `Room_Type`, `Order_Date` (date slicers get a handy range-slider format automatically).
3. Under **Format → Slicer settings**, choose style: **List**, **Dropdown**, or **Between** (for numeric/date ranges).
4. Use **Edit interactions** (Format tab → Edit Interactions) to control which visuals each slicer filters, if you don't want it to affect everything on the page.

### Step 6: Dashboard design principles (this is graded separately — don't skip it)

- **Top-left to bottom-right reading order**: put your most important KPIs (Cards) at the top, detail charts below.
- **Consistent color palette**: pick 3–4 colors and reuse them for the same category across all charts (e.g., always the same color for "Cash" payment method).
- **Avoid 3D charts and unnecessary pie charts with many slices** — they're hard to read.
- **Titles on every visual** stating what it shows.
- **Whitespace/alignment**: align visuals to a grid; don't overcrowd a page — Power BI has a **View → Gridlines & Snap to grid** option to help.
- **One clear insight per visual** — if a chart needs a paragraph to explain, it's the wrong chart.
- **Use tooltips** (Format → Tooltip) to add extra detail without cluttering the visual itself.

---

## Quick End-to-End Checklist (use this for every set)

- [ ] Load data, inspect with `.info()` / `.describe()`
- [ ] Standardize placeholder missing values → real `NaN`
- [ ] Fill/drop missing values with a justified strategy per column
- [ ] Remove duplicate rows (full-row and key-column)
- [ ] Standardize categorical text (case, whitespace, synonym mapping)
- [ ] Detect and handle outliers (domain rule or IQR)
- [ ] Validate financial/derived columns against their formula
- [ ] Convert date/time columns properly (Set-3, Set-4)
- [ ] Plot at least one histogram, one boxplot, and one bar/scatter to prove cleaning worked
- [ ] Export to CSV with `index=False`
- [ ] Load into Power BI, fix column types in Power Query
- [ ] Write DAX measures (distinguish `DISTINCTCOUNT` vs `SUM` vs `AVERAGE` correctly)
- [ ] Build Card visuals for KPIs
- [ ] Match chart type to the analysis question (see table above)
- [ ] Add slicers
- [ ] Apply layout/design principles before submitting
