# Machine Learning Guide (Part 3): Sample Solution on Set 1 (Hospital)

This guide walks through every ML step in your syllabus using the **cleaned Set 1 files** from Q1.
The same steps work for any set: only the column names and the target change.

**Syllabus checklist**

1. Identify input features and the target variable
2. Prepare the data for machine learning
3. Split into training and testing sets
4. Train the models
5. Generate predictions
6. Evaluate model performance
7. Classification and regression, with Decision Tree, Random Forest, Linear Regression, Logistic Regression, KNN and SVM

---

## 0. Setup (do this once, while online)

```
pip install scikit-learn
```

`scikit-learn` is the library that holds all the algorithms. After it is installed it works offline. Test it:

```python
import sklearn
print(sklearn.__version__)
```

Keep `patients_clean.csv`, `visits_clean.csv` and `billing_clean.csv` (the Q1 outputs) in the same folder as the new notebook.

---

## 1. The big idea in 6 lines

| Term | Meaning | Example (Set 1) |
|---|---|---|
| **Features (X)** | The inputs used to predict | Age, Department, Length_of_Stay_Days |
| **Target (y)** | The thing you want to predict | Admitted or not / Total_Bill |
| **Train set** | Data the model learns from (80%) | |
| **Test set** | Unseen data used to judge it (20%) | |
| **Classification** | Target is a **category** | Admitted / Not admitted |
| **Regression** | Target is a **number** | Total_Bill in rupees |

**Which algorithm does what**

| Algorithm | Classification | Regression | Needs scaling? |
|---|---|---|---|
| Logistic Regression | Yes (despite the name) | No | Yes |
| Linear Regression | No | Yes | Recommended |
| Decision Tree | Yes | Yes | No |
| Random Forest | Yes | Yes | No |
| KNN | Yes | Yes | **Yes** |
| SVM | Yes (`SVC`) | Yes (`SVR`) | **Yes** |

Remember: **Logistic Regression is for classification, Linear Regression is for regression.**

---

## 2. Load and combine the cleaned data

ML needs one table, so join the three files into one.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pat  = pd.read_csv('patients_clean.csv')
vis  = pd.read_csv('visits_clean.csv')
bill = pd.read_csv('billing_clean.csv')

df = vis.merge(pat, on='Patient_ID', how='left').merge(bill, on='Visit_ID', how='left')
print(df.shape)
df.head()
```

Check there are no gaps left:

```python
print(df.isna().sum())
```

If some columns show missing values (for example a visit whose patient was dropped in cleaning), drop those rows:

```python
df = df.dropna()
print(df.shape)
```

---

## 3. Identify features and targets

We build **two** problems from the same table.

| Problem | Target (y) | Why it makes sense |
|---|---|---|
| **Classification** | `Admitted` (1 = admitted, 0 = not) | Predict whether a visit leads to admission |
| **Regression** | `Total_Bill` | Predict the bill amount of a visit |

Create the classification target from the text column:

```python
df['Admitted'] = (df['Admission_Status'] == 'Admitted').astype(int)
print(df['Admitted'].value_counts())
```

Choose the features:

```python
# Classification features
cat_cols = ['Department', 'Visit_Type', 'Gender', 'City', 'Payment_Mode']   # text columns
num_cols = ['Age', 'Satisfaction_Score', 'Treatment_Cost']                   # number columns

# Regression features
reg_cat = ['Department', 'Visit_Type']
reg_num = ['Room_Charges', 'Length_of_Stay_Days', 'Age', 'Treatment_Cost']
```

**Rules for choosing features**

- Never include the target (or a column calculated from it) as a feature. That is called **data leakage**. For example, don't use `Net_Amount` to predict `Total_Bill`, because `Net_Amount` is built from it.
- Drop ID and name columns (`Patient_ID`, `Visit_ID`, `Patient_Name`). They carry no pattern.

---

## 4. Data preparation

Models understand only numbers, so there are two jobs: **encode text** and **scale numbers**.

### 4a. Encode text columns (one-hot encoding)

```python
X_clf = pd.get_dummies(df[cat_cols + num_cols], columns=cat_cols, drop_first=True)
y_clf = df['Admitted']

X_reg = pd.get_dummies(df[reg_cat + reg_num], columns=reg_cat, drop_first=True)
y_reg = df['Total_Bill']

print(X_clf.shape, X_reg.shape)
X_clf.head()
```

`get_dummies` turns a column like Department into several 0/1 columns (`Department_Neurology` and so on). `drop_first=True` removes one redundant column.

### 4b. Scale numbers (needed for Logistic Regression, KNN and SVM)

Scaling happens **after** the split (next step), so the test data stays unseen. Do it in the code at step 6.

---

## 5. Train and test split

```python
from sklearn.model_selection import train_test_split

# classification split (stratify keeps the same admitted/not ratio in both parts)
Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

# regression split
Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

print('Classification:', Xc_train.shape, Xc_test.shape)
print('Regression    :', Xr_train.shape, Xr_test.shape)
```

- `test_size=0.2` means 20% of rows are kept aside for testing.
- `random_state=42` makes the split the same every run, so your results repeat.

Now scale (fit on train only, then apply to both):

```python
from sklearn.preprocessing import StandardScaler

sc_c = StandardScaler()
Xc_train_s = sc_c.fit_transform(Xc_train)
Xc_test_s  = sc_c.transform(Xc_test)

sc_r = StandardScaler()
Xr_train_s = sc_r.fit_transform(Xr_train)
Xr_test_s  = sc_r.transform(Xr_test)
```

---

## 6. Classification: train, predict, evaluate

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# (model, uses scaled data?)
clf_models = {
    'Logistic Regression': (LogisticRegression(max_iter=1000), True),
    'Decision Tree':       (DecisionTreeClassifier(max_depth=5, random_state=42), False),
    'Random Forest':       (RandomForestClassifier(n_estimators=100, random_state=42), False),
    'KNN':                 (KNeighborsClassifier(n_neighbors=5), True),
    'SVM':                 (SVC(kernel='rbf'), True),
}

results = []
clf_preds = {}
for name, (model, scaled) in clf_models.items():
    Xtr, Xte = (Xc_train_s, Xc_test_s) if scaled else (Xc_train, Xc_test)
    model.fit(Xtr, yc_train)                 # TRAIN
    pred = model.predict(Xte)                # PREDICT
    clf_preds[name] = pred
    results.append({
        'Model': name,
        'Train Acc': round(model.score(Xtr, yc_train), 3),
        'Test Acc':  round(accuracy_score(yc_test, pred), 3),
        'Precision': round(precision_score(yc_test, pred), 3),
        'Recall':    round(recall_score(yc_test, pred), 3),
        'F1':        round(f1_score(yc_test, pred), 3),
    })

clf_results = pd.DataFrame(results)
clf_results
```

The three steps for every model are always **`fit` (train), `predict`, then compare with the real answers**.

### Detailed evaluation of one model

```python
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

best = 'Random Forest'
print(classification_report(yc_test, clf_preds[best]))

cm = confusion_matrix(yc_test, clf_preds[best])
ConfusionMatrixDisplay(cm, display_labels=['Not Admitted', 'Admitted']).plot(cmap='Blues')
plt.title(f'Confusion Matrix: {best}')
plt.show()
```

**Reading the numbers**

| Metric | Meaning |
|---|---|
| Accuracy | % of all predictions that were right |
| Precision | Of those predicted "Admitted", how many truly were |
| Recall | Of those truly "Admitted", how many the model found |
| F1 | Balance of precision and recall |
| Confusion matrix | Rows = actual, columns = predicted. The diagonal is correct |

**Overfitting check:** if Train Acc is far above Test Acc (for example 1.00 against 0.55), the model memorised the training data. Decision Tree and Random Forest do this often. Limit it with `max_depth`.

---

## 7. Regression: train, predict, evaluate

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

reg_models = {
    'Linear Regression': (LinearRegression(), True),
    'Decision Tree':     (DecisionTreeRegressor(max_depth=5, random_state=42), False),
    'Random Forest':     (RandomForestRegressor(n_estimators=100, random_state=42), False),
    'KNN':               (KNeighborsRegressor(n_neighbors=5), True),
    'SVR':               (SVR(kernel='rbf', C=1000), True),
}

results = []
reg_preds = {}
for name, (model, scaled) in reg_models.items():
    Xtr, Xte = (Xr_train_s, Xr_test_s) if scaled else (Xr_train, Xr_test)
    model.fit(Xtr, yr_train)                 # TRAIN
    pred = model.predict(Xte)                # PREDICT
    reg_preds[name] = pred
    results.append({
        'Model': name,
        'MAE':  round(mean_absolute_error(yr_test, pred), 0),
        'RMSE': round(np.sqrt(mean_squared_error(yr_test, pred)), 0),
        'R2 (test)':  round(r2_score(yr_test, pred), 3),
        'R2 (train)': round(model.score(Xtr, yr_train), 3),
    })

reg_results = pd.DataFrame(results)
reg_results
```

**Reading the numbers**

| Metric | Meaning | Good value |
|---|---|---|
| MAE | Average size of the error, in rupees | Lower is better |
| RMSE | Like MAE but punishes big errors more | Lower is better |
| R² | Share of the bill's variation the model explains | Closer to 1 is better (0 means no better than guessing the average, negative means worse) |

`SVR` ignores small errors by default, so with a target in the tens of thousands it needs a larger `C` (the penalty setting) than its default of 1. Try `C=1000`. If SVR still gives a negative R², try another `C` (for example 100, 1000 or 10000): too small makes it predict a flat average, too large makes it overfit.

---

## 8. Visualise the predictions

### Actual vs predicted (regression)

```python
best_r = 'Random Forest'
plt.figure(figsize=(6, 5))
plt.scatter(yr_test, reg_preds[best_r], alpha=0.5)
lims = [yr_test.min(), yr_test.max()]
plt.plot(lims, lims, 'r--', label='Perfect prediction')
plt.xlabel('Actual Total_Bill')
plt.ylabel('Predicted Total_Bill')
plt.title(f'Actual vs Predicted: {best_r}')
plt.legend()
plt.show()
```

Points close to the red line mean good predictions.

### Compare all models

```python
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
clf_results.set_index('Model')['Test Acc'].plot(kind='bar', ax=ax[0], ylim=(0, 1))
ax[0].set_title('Classification: Test Accuracy')
reg_results.set_index('Model')['R2 (test)'].plot(kind='bar', ax=ax[1])
ax[1].set_title('Regression: Test R2')
plt.tight_layout()
plt.show()
```

### Which features matter (Random Forest)

```python
rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(Xr_train, yr_train)
imp = pd.Series(rf.feature_importances_, index=X_reg.columns).sort_values()
imp.plot(kind='barh', figsize=(6, 4), title='Feature Importance (Random Forest)')
plt.show()
```

### Draw the Decision Tree

```python
from sklearn.tree import plot_tree

dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(Xc_train, yc_train)
plt.figure(figsize=(16, 7))
plot_tree(dt, feature_names=X_clf.columns, class_names=['Not Admitted', 'Admitted'], filled=True)
plt.show()
```

---

## 9. Predict for a new patient (shows you understand prediction)

```python
# take one row from the test set and predict it
sample = Xr_test.iloc[[0]]
print('Predicted bill :', rf.predict(sample)[0].round(0))
print('Actual bill    :', yr_test.iloc[0])
```

---

## 10. Honest reading of results

The practice data in this folder is **randomly generated**, so models will show **low accuracy (around 50 to 60%) and low R² (0 to 0.2)**. That is expected: there is no real pattern to find. In the exam the data may have clearer patterns. Always write what you actually see:

- If **Test Acc is close to Train Acc**, the model generalises well.
- If **Train is much higher than Test**, it is overfitting.
- If every model scores about the same, the features are weak predictors.
- The best model is the one with the **highest Test accuracy / R² and the lowest error**, not the highest Train score.

### Answer template for the exam

> *Target:* Admitted (classification) / Total_Bill (regression).
> *Features:* Age, Department, Visit_Type, Treatment_Cost, etc. IDs and names were dropped.
> *Preparation:* text columns one-hot encoded, numeric columns scaled with StandardScaler for Logistic Regression, KNN and SVM.
> *Split:* 80% train and 20% test, `random_state=42`.
> *Result:* Model X gave the best test accuracy of __ / R² of __, so it was chosen.

---

## 11. Common errors and fixes

| Error | Cause | Fix |
|---|---|---|
| `ValueError: could not convert string to float` | A text column went into the model | Use `pd.get_dummies` |
| `ValueError: Input contains NaN` | Missing values remain | `df = df.dropna()` or fill them |
| `ModuleNotFoundError: sklearn` | Library not installed | `pip install scikit-learn` |
| `ConvergenceWarning` (Logistic Regression) | Needs more iterations | `LogisticRegression(max_iter=1000)` |
| Very negative R² for SVR or KNN | Features not scaled, or SVR `C` badly chosen | Use the scaled arrays and try `C=100`, `1000` or `10000` |
| Test accuracy of exactly the share of one class | Model predicts only the majority class | Check the `confusion_matrix` |

## 12. One-page cheat sheet

```python
# 1. features and target
X = pd.get_dummies(df[features], drop_first=True);  y = df['target']
# 2. split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 3. scale (for Logistic Regression / KNN / SVM)
sc = StandardScaler();  X_train_s = sc.fit_transform(X_train);  X_test_s = sc.transform(X_test)
# 4. train
model.fit(X_train, y_train)
# 5. predict
pred = model.predict(X_test)
# 6. evaluate
accuracy_score(y_test, pred)   # classification
r2_score(y_test, pred)         # regression
```

| Model | Classification class | Regression class |
|---|---|---|
| Logistic / Linear | `LogisticRegression` | `LinearRegression` |
| Decision Tree | `DecisionTreeClassifier` | `DecisionTreeRegressor` |
| Random Forest | `RandomForestClassifier` | `RandomForestRegressor` |
| KNN | `KNeighborsClassifier` | `KNeighborsRegressor` |
| SVM | `SVC` | `SVR` |
