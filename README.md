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


# Naive Bayes From Scratch

A minimal implementation of both Multinomial and Gaussian Naïve Bayes classifiers in Python, along with a simple example using the Iris dataset.

---

## Files

- **`naive_bayes.py`**  
  Contains two classes:
  - **`MultinomialNB`**  
    - Designed for nonnegative integer “count” features (e.g., word counts).  
    - Uses Laplace smoothing (parameter `alpha`) to estimate P(x_j | y).  
    - Method `fit(X, y)` learns class priors and feature‐count probabilities.  
    - Method `predict(X)` returns the class with the highest joint log‐likelihood for each sample.  
  - **`GaussianNB`**  
    - Designed for continuous real‐valued features (assumes a Gaussian distribution per class).  
    - Uses `theta_` (mean) and `sigma_` (variance + smoothing) per feature‐class.  
    - Method `fit(X, y)` computes per‐class mean/variance and class priors.  
    - Method `predict(X)` returns the class that maximizes log P(y) + Σ log Gaussian(x|μ,σ).

  At the bottom of `naive_bayes.py` is an example under `if __name__ == "__main__":`:
  1. Loads Iris via `sklearn.datasets.load_iris()`  
  2. Splits into train/test (`20%` test, `random_state=0`)  
  3. Rounds continuous features to ints and trains `MultinomialNB(alpha=1.0)` (only for demonstration)  
  4. Trains `GaussianNB(var_smoothing=1e-9)` on the original continuous data  
  5. Prints accuracy for both models

---

## Dependencies

- **Python 3.6+**  
- [NumPy](https://numpy.org/)  
- [scikit-learn](https://scikit-learn.org/) (for data loading, train/test split, and accuracy metric)  

Install with:
```bash
pip install numpy scikit-learn
```
the output should look like:
---------------------------------
---------------------------------
MultinomialNB accuracy (rounded counts): 0.7
GaussianNB accuracy: 0.9666666666666667
---------------------------------
---------------------------------


# Discussion

Below is a brief comparison of the three “from-scratch” classifiers—kNN, Decision Tree, and Naïve Bayes—outlining their strengths, weaknesses, and best-use scenarios.

---

## k-Nearest Neighbors (kNN)

- **How it works (recap)**  
  Stores all training examples in memory. To classify a new point, it computes the distance (e.g., Euclidean) to every training sample, selects the k closest neighbors, and returns the majority label (or a distance-weighted vote).

- **Strengths**  
  - **No training phase**: fitting is just “store data,” so implementation is very simple.  
  - **Flexibility**: works for multiclass out of the box, and you can swap distance metrics (Euclidean, Manhattan, etc.).  
  - **Naturally nonparametric**: decision boundaries adapt to the local density of data.

- **Weaknesses**  
  - **Prediction cost**: O(n_train × n_features) per query, which can be slow for large datasets.  
  - **Feature scaling required**: if features are on different scales, some will dominate distance.  
  - **Sensitive to noise & irrelevant features**: a single outlier or high-dimensional sparse data can degrade performance.

- **Best use-cases**  
  - **Small-to-medium datasets** where storing all points and computing pairwise distances is tractable.  
  - **Low-dimensional numeric data** where distance is meaningful (e.g., simple pattern recognition, anomaly detection).  
  - When you want a *lazy learner*—no need to worry about overfitting a parametric model or tuning many hyperparameters (aside from k).

---

## Decision Tree

- **How it works (recap)**  
  Recursively partitions the feature space by selecting the feature and threshold that maximize information gain (entropy or Gini) at each node. Leaves are pure (single-class) or stop when a depth/size constraint is met. Prediction follows the path from root to leaf based on feature comparisons.

- **Strengths**  
  - **Interpretable**: once built, the tree can be printed or visualized, showing exactly which feature splits led to each decision.  
  - **Handles both numeric and categorical features** (with slight modification).  
  - **No feature scaling needed**: splits rely on ordering, not absolute distances.  
  - **Fast inference**: traversing a tree is O(depth) per sample, which is very efficient for wide datasets.

- **Weaknesses**  
  - **Prone to overfitting** if grown deep without pruning (can memorize training noise).  
  - **Unstable**: small changes in data can produce a very different tree.  
  - Needs explicit stopping criteria (max depth, min samples) or pruning to generalize well.

- **Best use-cases**  
  - **Medium-sized datasets** where interpretability is important (e.g., medical decision-making, finance).  
  - **Mixed feature types**: numeric and categorical.  
  - When you want a clear set of “if-then” rules and don’t mind tuning depth or pruning parameters.

---

## Naïve Bayes

- **How it works (recap)**  
  Assumes features are conditionally independent given the class.  
  - **MultinomialNB**: appropriate for discrete count data (e.g., word frequencies). Computes class priors and per-class, per-feature conditional probabilities with Laplace smoothing.  
  - **GaussianNB**: appropriate for continuous numeric data that roughly follow a Gaussian distribution in each class. Estimates per-class mean and variance and applies the Gaussian likelihood.

- **Strengths**  
  - **Very fast training and prediction**: just compute counts or means/variances. Suitable for large datasets.  
  - **Robust to irrelevant features**: since it multiplies independent likelihoods, an uninformative feature has minimal impact.  
  - **Works well with high-dimensional sparse data** (especially MultinomialNB on text): can handle thousands of features (vocabulary terms) efficiently.

- **Weaknesses**  
  - **Conditional independence assumption is often violated**: if features are strongly correlated, the model’s probability estimates can be poor.  
  - **Distribution mismatch**:  
    - MultinomialNB on non-count data or GaussianNB on heavily non-Gaussian data will underperform.  
    - Performance drops if the real feature distribution deviates sharply from the assumed model.

- **Best use-cases**  
  - **Text classification** (spam detection, sentiment analysis) with MultinomialNB or BernoulliNB (binary word presence).  
  - **Simple real-valued data** that roughly cluster per class in a bell curve (GaussianNB), e.g., iris measurements, certain sensor readings.  
  - **Baseline models**: even if assumptions are not perfect, Naïve Bayes often gives a quick, decent benchmark before trying more complex algorithms.

---

## Summary Comparison

| Model          | Training Cost      | Prediction Cost         | Interpretability     | Data Type                  | When to Choose                                           |
| -------------- | -------------------|-------------------------|----------------------|----------------------------|---------------------------------------------------------|
| **kNN**        | O(1) (store data)  | O(n_train × n_features) | Low (no explicit model) | Numeric, low-dimensional    | Small/medium data where distance is meaningful          |
| **Decision Tree** | O(n_train × n_features × log n_train) (for splits) | O(tree depth)            | High (clear rules)        | Numeric & categorical        | When interpretability and mixed features matter         |
| **MultinomialNB** | O(n_samples × n_features) | O(n_samples × n_features) | Medium (feature‐class probabilities) | Count data (text)           | Text classification or discrete count data              |
| **GaussianNB**   | O(n_samples × n_features) | O(n_samples × n_features) | Medium (means/variances per feature‐class) | Continuous numeric          | When features are roughly Gaussian within each class    |

- **kNN** is best for small, low-dimensional numeric tasks where you don’t mind paying the distance-computation cost at inference.  
- **Decision Tree** is ideal if you need an interpretable “if-then” model that handles mixed data types without scaling.  
- **MultinomialNB** excels on high-dimensional sparse count data (e.g., bag-of-words in NLP).  
- **GaussianNB** is a simple yet powerful choice when your continuous features approximate a normal distribution within each class and you want extremely fast training/prediction.  

Choose the model whose assumptions align best with your data’s structure and your deployment constraints (speed vs. interpretability vs. dataset size).
