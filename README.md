# kNN From Scratch

A minimal implementation of the k-Nearest Neighbors classifier in Python, accompanied by a short example script that:

1. Loads the Iris dataset  
2. Splits it into train/test  
3. Trains the custom `KNN` class (in `KNN.py`)  
4. Prints the predicted labels for the test set and the overall accuracy

---

## Files

- **`KNN.py`**  
  Contains a simple `KNN` class:
  - `fit(X, y)` stores the training data  
  - `predict(X)` computes Euclidean distances to find the k nearest neighbors and returns the majority‐vote label for each test sample

- **`dataset.py`**  
  A small driver script that:
  - Imports and splits the Iris dataset  
  - (Optionally) shows a quick scatter plot of petal length vs. petal width  
  - Instantiates `KNN(k=5)`, calls `fit(...)` and `predict(...)`  
  - Prints the array of predicted labels and the test‐set accuracy

---

## Usage

1. **Install dependencies** (Python 3.6+):
   ```bash
   pip install numpy scikit-learn matplotlib


From the project directory, run:
------------
python dataset.py
------------
You should see something like:
-----------------
[1 2 2 0 1 0 0 0 1 2 1 0 2 1 0 1 2 0 2 1 1 1 1 1 2 0 2 1 2 0]
Accuracy: 0.9666666666666667
-----------------

