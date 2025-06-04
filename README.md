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

# Decision Tree From Scratch

A minimal, from-scratch implementation of a Decision Tree classifier in Python, accompanied by a short script (`decision_tree.py`) that:

1. Reads the Iris dataset from `Iris.csv`  
2. Splits it into train/test  
3. Builds a custom `DecisionTree` (with entropy or Gini‐based splits)  
4. Prints the tree structure to the console  
5. Predicts on the test set and reports overall accuracy  

---

## Files

- **`decision_tree.py`**  
  Implements:
  - A `Node` class to represent each tree node (holding feature index, threshold, children, information gain, or leaf value)  
  - A `DecisionTree` class that:
    - **`build_tree(dataset)`**: finds the best feature/threshold split (maximizing information gain via entropy by default), recursively builds left/right subtrees, and creates leaf nodes when pure or stopping criteria are met  
    - **`fit(X, y)`**: concatenates features and labels, then calls `build_tree` to set `self.root`  
    - **`print_tree(tree)`**: prints a human-readable, indented view of every split (feature ≤ threshold with info gain) and leaf values  
    - **`predict(X)`**: walks each sample down from `self.root` to a leaf, returning the leaf’s majority‐class label  
    - At the bottom, the script:
      1. Loads `Iris.csv` into a DataFrame  
      2. Converts it to NumPy arrays for features (`X`) and labels (`Y`)  
      3. Splits with `train_test_split` (`test_size=0.2, random_state=41`)  
      4. Instantiates `DecisionTree(min_samples_split=2, max_depth=3)`  
      5. Calls `fit(X_train, y_train)` and then `print_tree()`  
      6. Uses `predict(X_test)` and prints accuracy via `sklearn.metrics.accuracy_score`

- **`Iris.csv`**  
  A CSV file containing the classic four‐feature Iris dataset plus a header column (no header in the file itself). The code in `decision_tree.py` uses:
  ```python
  col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
  data = pd.read_csv('Iris.csv', header=None, names=col_names)


the output should look like :
---------------
X_2 <= 1.9? 0.9264046681474138
 left:Iris-setosa
 right:X_3 <= 1.7? 0.6733101937922419
   left:X_2 <= 5.0? 0.21241631061007382
     left:X_0 <= 4.9? 0.11866093149667024
       left:Iris-versicolor
       right:Iris-versicolor
     right:X_0 <= 6.0? 0.8112781244591328
       left:Iris-versicolor
       right:Iris-virginica
   right:X_0 <= 7.9? 0.18717625687320813
     left:X_2 <= 4.8? 0.1326097254024287
       left:Iris-versicolor
       right:Iris-virginica
     right:Species
Accuracy: 0.9354838709677419
---------------