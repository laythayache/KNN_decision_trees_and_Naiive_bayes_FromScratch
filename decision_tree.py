import numpy as np
import pandas as pd

col_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
data = pd.read_csv('Iris.csv', header=None, names=col_names)
data.head(10)


class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature = feature  # Feature index to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.info_gain = info_gain  # Information gain from the split
        self.value = value  # Value for leaf node
        
class DecisionTree():
    def __init__(self, min_samples_split=2, max_depth=2):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
    def build_tree(self, dataset, curr_depth=0):
        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape
        
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split and best_split.get('info_gain', 0) > 0:
                left_subtree = self.build_tree(best_split['left'], curr_depth + 1)
                right_subtree = self.build_tree(best_split['right'], curr_depth + 1)
                return Node(feature=best_split['feature'], threshold=best_split['threshold'],
                            left=left_subtree, right=right_subtree, info_gain=best_split['info_gain'])
        leaf_value = self.calculate_leaf_value(Y)
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                left_subtree, right_subtree = self.split_dataset(dataset, feature_index, threshold)
                if len(left_subtree) > 0 and len(right_subtree) > 0:
                    y_left = left_subtree[:, -1]
                    y_right = right_subtree[:, -1]
                    info_gain = self.information_gain(dataset[:, -1], y_left, y_right)
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_split['feature'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['left'] = left_subtree
                        best_split['right'] = right_subtree
                        best_split['info_gain'] = info_gain
        return best_split
    
    
    def split_dataset(self, dataset, feature_index, threshold):
        left_subtree = dataset[dataset[:, feature_index] <= threshold]
        right_subtree = dataset[dataset[:, feature_index] > threshold]
        return left_subtree, right_subtree
    
    def information_gain(self,parent, l_child, r_child, mode='entropy'):
        weight_l = len(l_child)/ len(parent)
        weight_r = len(r_child)/ len(parent)
        if mode == 'gini':
            gain= self.gini_index(parent) - (weight_l * self.gini_index(l_child) + weight_r * self.gini_index(r_child))
        elif mode == 'entropy':
            gain = self.entropy(parent) - (weight_l * self.entropy(l_child) + weight_r * self.entropy(r_child))
        return gain
    
    def entropy(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        entropy = 0
        for count in counts:
            p_cls = count / len(y)
            if p_cls > 0:
                entropy -= p_cls * np.log2(p_cls)
        return entropy

    def gini_index(self, y):
        class_labels, counts = np.unique(y, return_counts=True)
        gini = 0
        for count in counts:
            p_cls = count / len(y)
            gini += p_cls ** 2
        return 1 - gini

    def calculate_leaf_value(self, Y):
        Y = list(Y)
        return max(Y, key=Y.count)
    
    def print_tree(self,tree = None, indent=" "):
        if not tree:
            tree = self.root
        if tree.value is not None:
            print(tree.value)
        else:
            print("X_" + str(tree.feature) + " <= " + str(tree.threshold) + "?",tree.info_gain)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + "  ")
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + "  ")
    
    def fit(self, X, y):
        dataset = np.concatenate((X, y), axis=1)
        self.root = self.build_tree(dataset)
    
    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return predictions
    
    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature]
        if feature_value <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)
        
        
X =data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1, 1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=41)

classifier = DecisionTree(min_samples_split=2, max_depth=3)
classifier.fit(X_train, y_train)
classifier.print_tree()
y_pred = classifier.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy:", accuracy_score(y_test, y_pred))