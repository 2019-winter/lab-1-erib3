---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Name(s)
**Ethan Ribera**


**Instructions:** This is an individual assignment, but you may discuss your code with your neighbors.


# Python and NumPy

While other IDEs exist for Python development and for data science related activities, one of the most popular environments is Jupyter Notebooks.

This lab is not intended to teach you everything you will use in this course. Instead, it is designed to give you exposure to some critical components from NumPy that we will rely upon routinely.

## Exercise 0
Please read and reference the following as your progress through this course. 

* [What is the Jupyter Notebook?](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/What%20is%20the%20Jupyter%20Notebook.ipynb#)
* [Notebook Tutorial](https://www.datacamp.com/community/tutorials/tutorial-jupyter-notebook)
* [Notebook Basics](https://nbviewer.jupyter.org/github/jupyter/notebook/blob/master/docs/source/examples/Notebook/Notebook%20Basics.ipynb)

**In the space provided below, what are three things that still remain unclear or need further explanation?**


**YOUR ANSWER HERE**


## Exercises 1-7
For the following exercises please read the Python appendix in the Marsland textbook and answer problems A.1-A.7 in the space provided below.


## Exercise 1

```python
# YOUR SOLUTION HERE
import numpy as np
r, c = 6, 4
a = np.array([[2 for x in range(c)] for y in range(r)])
a
#a2 = np.ones((6,4), dtype=int)*2
```

## Exercise 2

```python
# YOUR SOLUTION HERE
b = np.array([[1 for x in range(c)] for y in range(r)])
# b2 = np.ones((6,4))
b[range(4), range(4)] = 3
b
```

## Exercise 3

```python
# YOUR SOLUTION HERE
print(a * b)
# * works because element by element works
# np.dot(a,b) does not work because the matrices are not aligned
```

## Exercise 4

```python
# YOUR SOLUTION HERE
print(np.dot(a.transpose(), b))
print(np.dot(a, b.transpose()))

# why are these different shapes?? 
# because outer dimension is the size of matrix
```

## Exercise 5

```python
# YOUR SOLUTION HERE
def test_printing():
    print("hello")
test_printing()
```

## Exercise 6

```python
# YOUR SOLUTION HERE
def random_array():
    rows = np.random.randint(10)
    columns = np.random.randint(10)
    ar = np.array([[np.random.randint(10) for x in range(columns)] for y in range(rows)])
    #a = np.random.rand(6,4)
    print(ar)
    print("sum " + str(np.sum(ar)))
    print("mean " + str(np.mean(ar)))
    
random_array()
```

## Exercise 7

```python
# YOUR SOLUTION HERE
def count_ones(ar):
    ones = 0
    for r in range(ar.shape[0]):
        for c in range(ar.shape[1]):
            if ar[r][c] == 1:
                ones += 1
    return ones,len(np.where(b==1)[0])

print(count_ones(b))

```

## Excercises 8-???
While the Marsland book avoids using another popular package called Pandas, we will use it at times throughout this course. Please read and study [10 minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html) before proceeding to any of the exercises below.


## Exercise 8
Repeat exercise A.1 from Marsland, but create a Pandas DataFrame instead of a NumPy array.

```python
# YOUR SOLUTION HERE
import pandas as pd
import numpy as np
a = pd.DataFrame(np.ones((6,4))*2)
a

```

## Exercise 9
Repeat exercise A.2 using a DataFrame instead.

```python
# YOUR SOLUTION HERE
b = np.ones((6,4))
b = pd.DataFrame(b)
b.iloc[range(4), range(4)] = 3
b
```

## Exercise 10
Repeat exercise A.3 using DataFrames instead.

```python
# YOUR SOLUTION HERE
display(a*b)
# a*b does element by element whereas dot does matrix multiplication
```

## Exercise 11
Repeat exercise A.7 using a dataframe.

```python
# YOUR SOLUTION HERE
import pandas as pd

def count_ones2(ar):
    ones = 0
    for r in range(ar.shape[0]):
        for c in range(ar.shape[1]):
            if ar.iloc[r,c] == 1:
                ones += 1
    return ones,len(np.where(b==1)[0])
print(count_ones2(pd.DataFrame(b)))
```

## Exercises 12-14
Now let's look at a real dataset, and talk about ``.loc``. For this exercise, we will use the popular Titanic dataset from Kaggle. Here is some sample code to read it into a dataframe.

```python
titanic_df = pd.read_csv(
    "https://raw.githubusercontent.com/dlsun/data-science-book/master/data/titanic.csv"
)
titanic_df
```

Notice how we have nice headers and mixed datatypes? That is one of the reasons we might use Pandas. Please refresh your memory by looking at the 10 minutes to Pandas again, but then answer the following.


## Exercise 12
How do you select the ``name`` column without using .iloc?

```python
## YOUR SOLUTION HERE
titanic_df.name

```

## Exercise 13
After setting the index to ``sex``, how do you select all passengers that are ``female``? And how many female passengers are there?

```python
## YOUR SOLUTION HERE
titanic_df.set_index('sex',inplace=True)
titanic_df.loc['female']
# 466 female passengers
```

## Exercise 14
How do you reset the index?

```python
## YOUR SOLUTION HERE
```

```python
titanic_df.reset_index(inplace=True)
titanic_df
```

```python

```
