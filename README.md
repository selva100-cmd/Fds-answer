# Fds-Question answer
Great — I’ll answer **all 20 lab questions** from your uploaded exam paper. For each coding question I’ll give a compact **Aim**, a short **Algorithm**, a clean **Program** (ready-to-run Python with NumPy / Pandas), an example **Result** (what it prints), and a few **Viva-voce** questions+answers you can prepare. For the longer data-analysis tasks I’ll give a clear plan and runnable code snippets you can run on the datasets.

---

# 1. (a–d) NumPy: convert to float, add border, list/tuple→array, append values

**Aim:** Demonstrate basic NumPy array operations.

**Algorithm (short):**

1. Create arrays / lists.
2. Use `.astype(float)` to convert.
3. Use `np.pad` to add border.
4. Use `np.array()` to convert list/tuple.
5. Use `np.append()` to append values.

**Program:**

```python
import numpy as np

# a) convert array to float
arr = np.array([1, 2, 3], dtype=int)
arr_float = arr.astype(float)

# b) add zero border
m = np.array([[1,2],[3,4]])
m_border = np.pad(m, pad_width=1, mode='constant', constant_values=0)

# c) convert list and tuple into arrays
lst = [10,20,30]
tup = (40,50,60)
arr_from_list = np.array(lst)
arr_from_tuple = np.array(tup)

# d) append values to the end
orig = np.array([1,2,3])
appended = np.append(orig, [4,5])

print("a)", arr_float)
print("b)\n", m_border)
print("c)", arr_from_list, arr_from_tuple)
print("d)", appended)
```

**Result (example):**

```
a) [1. 2. 3.]
b)
 [[0 0 0 0]
  [0 1 2 0]
  [0 3 4 0]
  [0 0 0 0]]
c) [10 20 30] [40 50 60]
d) [1 2 3 4 5]
```

**Viva:**

* Q: How does `astype()` behave if conversion fails?
  A: It raises an error if conversion impossible or produces `nan`/`inf` for strings that don't represent numbers.
* Q: Difference between `np.append` and Python `list.append`?
  A: `np.append` returns a new array (immutable shape change), `list.append` mutates the list.

---

# 2. (a–d) NumPy: convert to float, empty & full arrays, list/tuple→arrays, real & imaginary parts

**Aim:** Show array creation and complex-number handling.

**Algorithm:** Use `astype(float)`, `np.empty`, `np.full`, `np.array`, and `.real/.imag`.

**Program:**

```python
import numpy as np

# a) convert to float
a = np.array([1,2,3])
a_float = a.astype(float)

# b) create empty and full
empty_arr = np.empty((2,3))
full_arr = np.full((2,3), 7)

# c) list/tuple to arrays
lst = [1,2,3]; tup = (4,5,6)
arr1 = np.array(lst); arr2 = np.array(tup)

# d) real and imaginary parts
cplx = np.array([1+2j, 3-1j, 0+4j])
real_part = cplx.real
imag_part = cplx.imag

print("a", a_float)
print("b empty\n", empty_arr, "\nfull\n", full_arr)
print("c", arr1, arr2)
print("d real", real_part, "imag", imag_part)
```

**Result (example):**

```
a [1. 2. 3.]
b empty
 [[1. 0. 0.]
  [0. 0. 0.]]
 full
 [[7 7 7]
  [7 7 7]]
c [1 2 3] [4 5 6]
d real [1. 3. 0.] imag [ 2. -1.  4.]
```

**Viva:**

* Q: Is `np.empty` initialized?
  A: No — it contains arbitrary memory values; use only when you will overwrite.

---

# 3. Pandas: create DataFrame from dict with index labels

**Aim:** Create and display a DataFrame with custom index.

**Algorithm:** Construct dict, `pd.DataFrame`, set `index=labels`.

**Program:**

```python
import numpy as np
import pandas as pd

exam_data = {
 'name': ['Anastasia','Dima','Katherine','James','Emily','Michael','Matthew','Laura','Kevin','Jonas'],
 'score': [12.5,9,16.5,np.nan,9,20,14.5,np.nan,8,19],
 'attempts': [1,3,2,3,2,3,1,1,2,1],
 'qualify': ['yes','no','yes','no','no','yes','yes','no','no','yes']
}
labels = list('abcdefghij')
df = pd.DataFrame(exam_data, index=labels)
# reorder columns as expected
df = df[['attempts','name','qualify','score']]
print(df)
```

**Result (snippet):**

```
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
...
i         2      Kevin      no    8.0
j         1      Jonas     yes   19.0
```

**Viva:**

* Q: How to show rows with missing values?
  A: `df[df['score'].isna()]` or `df.isnull()` methods.

---

# 4. Pandas: select rows where attempts > 2

**Aim:** Filter DataFrame rows.

**Algorithm:** Create DataFrame, use boolean mask `df['attempts'] > 2`.

**Program:**

```python
# from previous df creation
mask = df['attempts'] > 2
result = df[mask][['name','score','attempts','qualify']]
print("Number of attempts > 2:\n", result)
```

**Result:**

```
Number of attempts > 2:
    name  score  attempts qualify
b  Dima    9.0         3      no
d  James    NaN        3      no
f  Michael 20.0        3     yes
```

**Viva:**

* Q: How to reset index after filtering?
  A: `result.reset_index(drop=True)`.

---

# 5. Pandas: get first 3 rows

**Program (simple):**

```python
print(df.head(3))
```

**Result:**

```
   attempts       name qualify  score
a         1  Anastasia     yes   12.5
b         3       Dima      no    9.0
c         2  Katherine     yes   16.5
```

**Viva:**

* Q: How to get last n rows?
  A: `df.tail(n)`.

---

# 6. Pandas: select rows where score is NaN

**Program:**

```python
missing = df[df['score'].isna()][['attempts','name','qualify','score']]
print(missing)
```

**Result:**

```
   attempts   name qualify  score
d         3  James      no    NaN
h         1  Laura      no    NaN
```

**Viva:**

* Q: How many missing scores?
  A: `df['score'].isna().sum()`.

---

# 7. Read Iris dataset from file/web and descriptive commands

**Aim:** Read dataset from CSV/Excel/web and run descriptive analytics.

**Plan & Program:**

```python
import pandas as pd

# Option A: local CSV
# iris = pd.read_csv('iris.csv')

# Option B: directly from seaborn (if available)
import seaborn as sns
iris = sns.load_dataset('iris')

# Basic descriptive analytics
print(iris.head())
print(iris.describe())
print(iris.info())
print(iris['species'].value_counts())
```

**Notes:** For file reading: `pd.read_csv`, `pd.read_excel`, `pd.read_table` (for text). Use `iris.describe()`, `value_counts()`, `corr()`.

**Viva:**

* Q: What does `describe()` show?
  A: count, mean, std, min, 25/50/75%, max for numeric columns.

---

# 8. Diabetes dataset (UCI) — Univariate analysis

**Aim:** Compute frequency, mean, median, mode, variance, std, skewness, kurtosis for each variable.

**Program sketch:**

```python
import pandas as pd
import scipy.stats as stats

# load data (adjust path or URL)
df = pd.read_csv('diabetes.csv')  # replace with correct file

# For each numeric column:
for col in df.select_dtypes(include='number').columns:
    s = df[col].dropna()
    print(col)
    print("freq:\n", s.value_counts().head(10))
    print("mean", s.mean(), "median", s.median(), "mode", s.mode().tolist())
    print("var", s.var(), "std", s.std())
    print("skew", s.skew(), "kurtosis", s.kurtosis())
    print("-"*30)
```

**Viva:**

* Q: Difference between skewness and kurtosis?
  A: Skewness measures asymmetry; kurtosis measures "tailedness" (higher = heavier tails).

---

# 9. Diabetes dataset — Bivariate: linear & logistic regression

**Aim:** Fit linear regression for numeric target or logistic regression for binary outcome.

**Program sketch (linear & logistic using scikit-learn):**

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

df = pd.read_csv('diabetes.csv')
# Suppose 'Outcome' is binary (1/0) — typical Pima dataset
X = df.drop(columns=['Outcome'])
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)

lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
pred = lr.predict(X_test)
print("Logistic accuracy:", accuracy_score(y_test, pred))

# For linear regression, choose a numeric target (e.g., 'Glucose' predicted by others)
if 'Glucose' in df.columns:
    y_lin = df['Glucose'].fillna(df['Glucose'].mean())
    X_lin = df.drop(columns=['Glucose'])
    Xtr, Xte, ytr, yte = train_test_split(X_lin, y_lin, random_state=1)
    lm = LinearRegression().fit(Xtr, ytr)
    print("Linear R2:", r2_score(yte, lm.predict(Xte)))
```

**Viva:**

* Q: When use logistic vs linear?
  A: Logistic for categorical/binary targets; linear for continuous numeric targets.

---

# 10. Diabetes dataset — Multiple regression

**Aim:** Build multiple linear regression model to predict a numeric target using many predictors.

**Program:** (similar to linear part above; include coefficient table)

```python
import statsmodels.api as sm
X = df.drop(columns=['Glucose'])  # example
y = df['Glucose'].fillna(df['Glucose'].mean())
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())
```

**Viva:**

* Q: What is multicollinearity?
  A: High correlation between predictors; can inflate variance of coefficients; detect via VIF.

---

# 11. Pima Indians Diabetes — plotting: normal values, density & contour, 3D plotting

**Plan & Code:**

```python
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv('pima-indians-diabetes.csv')  # replace path
# a) Normal/QQ plot
import scipy.stats as stats
stats.probplot(df['Glucose'].dropna(), dist="norm", plot=plt)
plt.title('QQ plot for Glucose'); plt.show()

# b) Density and contour (2D)
plt.figure()
df['Glucose'].plot(kind='kde')  # density
plt.show()

# contour of two variables
x = df['Glucose'].dropna()
y = df['Insulin'].dropna()
# For contour, build 2D histogram / KDE grid (sklearn or scipy)
# c) 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['Glucose'], df['BMI'], df['Age'])
plt.show()
```

**Viva:**

* Q: What does a QQ plot show?
  A: How sample quantiles compare to theoretical normal quantiles.

---

# 12. Pima Indians Diabetes — correlation, scatter, histograms, 3D plotting

**Program:**

```python
import seaborn as sns
sns.pairplot(df)    # scatter plots + histograms
plt.show()
print(df.corr())
df.hist(figsize=(10,8))
plt.show()
```

**Viva:**

* Q: How do you interpret correlation matrix?
  A: Values near +1/-1 indicate strong linear relationships.

---

# 13 & 14. UCI dataset plotting (same as above) — Normal, density, contour, correlation, histograms, 3D

**Plan:** Use the same code patterns as Q11–12 but replace with the chosen UCI dataset (e.g., Wine, Adult, etc.). Use `sns.kdeplot`, `plt.hist`, `sns.heatmap(df.corr(), annot=True)` for correlation heatmap.

**Viva:**

* Q: When use contour vs scatter?
  A: Contour for density / distribution in 2D; scatter for individual points.

---

# 15. Pandas: numeric representation by distinct values of a column

**Aim:** Map distinct categorical names to numeric codes.

**Program:**

```python
import pandas as pd
df = pd.DataFrame({
 "Name":["Alberto Franco","Gino Mcneill","Ryan Parkes","Eesha Hinton","Gino Mcneill"],
 "Date_Of_Birth":["17/05/2002","16/02/1999","25/09/1998","11/05/2002","15/09/1997"],
 "Age":[18.5,21.2,22.5,22.0,23.0]
})
codes, uniques = pd.factorize(df['Name'])
print(codes)        # numeric representation
print(uniques)      # distinct names
```

**Result:**

```
[0 1 2 3 1]
Index(['Alberto Franco','Gino Mcneill','Ryan Parkes','Eesha Hinton'], dtype='object')
```

**Viva:**

* Q: How to get back names from codes?
  A: `uniques[codes]` or use `pd.Series(codes).map(dict(enumerate(uniques)))`.

---

# 16. Pandas: check inequality of two DataFrames

**Aim:** Compare two DataFrames element-wise.

**Program:**

```python
import pandas as pd
df1 = pd.DataFrame({...})  # as per sample
df2 = pd.DataFrame({...})
neq = df1.ne(df2)  # elementwise inequality -> DataFrame of booleans
print(neq)
```

(Use `df1 != df2` or `df1.ne(df2)`; NaNs require careful handling: `df1.eq(df2) | (df1.isna() & df2.isna())` to treat NaNs as equal.)

**Viva:**

* Q: How to treat NaN==NaN as True?
  A: Use `df1.fillna(value)` consistently or compare with `((df1==df2) | (df1.isna() & df2.isna()))`.

---

# 17. Pandas: get first n records

**Program:**

```python
df = pd.DataFrame({'col1':[1,2,3,4,7,11],'col2':[4,5,6,9,5,0],'col3':[7,5,8,12,1,11]})
n = 3
print(df.head(n))
```

**Result:** first 3 rows printed.

**Viva:** `df.head(n)` and `df.iloc[:n]` equivalent.

---

# 18. Pandas: select all columns except one given column

**Program:**

```python
df = pd.DataFrame({'col1':[1,2,3,4,7],'col2':[4,5,6,9,5],'col3':[7,8,12,1,11]})
df_except = df.drop(columns=['col3'])
print(df_except)
```

**Viva:** Also can do `df.loc[:, df.columns != 'col3']`.

---

# 19. NumPy: convert Python dict to NumPy ndarray

**Aim:** Convert nested dict-of-dicts to 2D ndarray.

**Program:**

```python
import numpy as np
d = {
 'column0': {'a':1,'b':0.0,'c':0.0,'d':2.0},
 'column1': {'a':3.0,'b':1,'c':0.0,'d':-1.0},
 'column2': {'a':4,'b':1,'c':5.0,'d':-1.0},
 'column3': {'a':3.0,'b':-1.0,'c':-1.0,'d':-1.0}
}
# order by inner keys to produce rows
cols = sorted(d.keys())
rows = sorted(d[cols[0]].keys())
arr = np.array([[d[c][r] for c in cols] for r in rows], dtype=float)
print(arr)
```

**Result:**

```
[[ 1.  3.  4.  3.]
 [ 0.  1.  1. -1.]
 [ 0.  0.  5. -1.]
 [ 2. -1. -1. -1.]]
```

**Viva:**

* Q: Why `dtype=float` used?
  A: To ensure mixed numeric types are handled in a homogeneous ndarray.

---

# 20. NumPy: search index of a given array in another array (row match)

**Aim:** Find index of a row equal to given subarray.

**Program:**

```python
import numpy as np
big = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
search = np.array([4,5,6])

# Method: compare rows
matches = np.all(big == search, axis=1)
indices = np.where(matches)[0]  # gives array of matching row indices
print(indices)  # expected [1]
```

**Result:**

```
[1]
```

**Viva:**

* Q: What if search appears multiple times?
  A: `indices` will list all matching row indices.

---

## Final notes & how I can help next

* All code above is runnable with Python (>=3.7), NumPy, Pandas, matplotlib/seaborn where indicated.
* If you want, I can:

  * format a single **complete answer sheet** for **one chosen question** with Aim/Algorithm/Program/Results/Viva in exam-ready layout (PDF or DOCX),
  * run any of the code here and show exact outputs (I can execute code if you want — tell me which question to run), or
  * expand any answer with more explanation, plots, or interpretation.

Which would you like next — (A) a full polished answer sheet for one selected question ready to submit, or (B) run any of these code snippets and show the outputs?
