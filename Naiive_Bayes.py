import numpy as np


class MultinomialNB:
    """
    Multinomial Naïve Bayes classifier from scratch.
    Suitable for count-based features (e.g., word counts in text).
    """

    def __init__(self, alpha=1.0):
        """
        alpha: Laplace smoothing parameter (≥ 0).
        """
        self.alpha = alpha
        self.class_log_prior_ = None      # Log P(y=c)
        self.feature_log_prob_ = None     # Log P(x_j | y=c)
        self.classes_ = None              # Unique class labels
        self.n_features_ = None           # Number of features

    def fit(self, X, y):
        """
        Fit the MultinomialNB model.
        
        Parameters:
        - X: array-like of shape (n_samples, n_features), non-negative counts
        - y: array-like of shape (n_samples,), class labels
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features
        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = self.classes_.shape[0]

        # Compute log prior: log P(y=c) = log(count_c / n_samples)
        self.class_log_prior_ = np.log(class_counts / n_samples)

        # Initialize feature count matrix: shape (n_classes, n_features)
        smoothed_fc = np.zeros((n_classes, n_features), dtype=np.float64)
        smoothed_cc = np.zeros(n_classes, dtype=np.float64)  # total count per class

        # For each class, sum up counts of each feature
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            # Sum counts of each feature for class c
            class_feature_count = X_c.sum(axis=0)
            smoothed_fc[idx, :] = class_feature_count + self.alpha
            # Total count of all features for class c (with smoothing)
            smoothed_cc[idx] = smoothed_fc[idx, :].sum()

        # Compute log P(x_j | y=c) = log((count_{c,j} + alpha) / (total_count_c + alpha * n_features))
        # But since we already added alpha to each feature count, denominator is smoothed_cc[idx] + alpha * 0
        self.feature_log_prob_ = np.log(smoothed_fc / smoothed_cc.reshape(-1, 1))

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        - X: array-like of shape (n_samples, n_features), non-negative counts
        
        Returns:
        - array of shape (n_samples,), predicted class labels
        """
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape
        if n_features != self.n_features_:
            raise ValueError(f"Number of features in input ({n_features}) does not match training data ({self.n_features_}).")

        # For each sample x, compute for each class: log P(y=c) + sum_j x_j * log P(x_j | y=c)
        jll = []  # joint log-likelihood: shape (n_samples, n_classes)
        for x in X:
            # Compute x * log P(x_j | y=c) summed over features
            # shape: (n_classes,)
            log_likelihood = (x @ self.feature_log_prob_.T)
            log_prior = self.class_log_prior_
            jll.append(log_prior + log_likelihood)

        jll = np.array(jll)  # shape (n_samples, n_classes)
        # Pick the class with highest joint log-likelihood
        class_indices = np.argmax(jll, axis=1)
        return self.classes_[class_indices]


class GaussianNB:
    """
    Gaussian Naïve Bayes classifier from scratch.
    Suitable for continuous features (assumes each feature ~ Gaussian per class).
    """

    def __init__(self, var_smoothing=1e-9):
        """
        var_smoothing: small number added to variances to avoid division by zero
        """
        self.var_smoothing = var_smoothing
        self.classes_ = None                # Unique class labels
        self.class_count_ = None            # Number of samples per class
        self.class_log_prior_ = None        # Log P(y=c)
        self.theta_ = None                  # Mean of each feature per class (shape: n_classes x n_features)
        self.sigma_ = None                  # Variance of each feature per class (shape: n_classes x n_features)
        self.n_features_ = None             # Number of features

    def fit(self, X, y):
        """
        Fit the GaussianNB model.
        
        Parameters:
        - X: array-like of shape (n_samples, n_features), continuous features
        - y: array-like of shape (n_samples,), class labels
        """
        X = np.array(X, dtype=np.float64)
        y = np.array(y)
        n_samples, n_features = X.shape
        self.n_features_ = n_features

        self.classes_, class_counts = np.unique(y, return_counts=True)
        n_classes = self.classes_.shape[0]
        self.class_count_ = class_counts

        # Compute log prior: log(count_c / n_samples)
        self.class_log_prior_ = np.log(class_counts / n_samples)

        # Initialize arrays for mean and variance
        self.theta_ = np.zeros((n_classes, n_features), dtype=np.float64)
        self.sigma_ = np.zeros((n_classes, n_features), dtype=np.float64)

        # Compute mean and variance for each class
        for idx, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.theta_[idx, :] = X_c.mean(axis=0)
            # Use unbiased estimator (ddof=0) for variance, then add var_smoothing
            self.sigma_[idx, :] = X_c.var(axis=0) + self.var_smoothing

    def _gaussian_log_prob(self, class_idx, x):
        """
        Compute the log probability of sample x under class class_idx:
        log P(x | y=class_idx) = sum over features of log Gaussian(x_j; mu, var)
        """
        mean = self.theta_[class_idx]
        var = self.sigma_[class_idx]
        # Compute per-feature log-probabilities:
        # log[1 / sqrt(2π var)] - ((x - mean)^2 / (2 var))
        log_coeff = -0.5 * np.log(2.0 * np.pi * var)
        log_exp = -0.5 * ((x - mean) ** 2) / var
        return np.sum(log_coeff + log_exp)

    def predict(self, X):
        """
        Predict class labels for samples in X.
        
        Parameters:
        - X: array-like of shape (n_samples, n_features)
        
        Returns:
        - array of shape (n_samples,), predicted class labels
        """
        X = np.array(X, dtype=np.float64)
        n_samples, n_features = X.shape
        if n_features != self.n_features_:
            raise ValueError(f"Number of features in input ({n_features}) does not match training data ({self.n_features_}).")

        n_classes = self.classes_.shape[0]
        jll = np.zeros((n_samples, n_classes), dtype=np.float64)  # joint log-likelihood

        for idx, c in enumerate(self.classes_):
            # For each class, compute log prior + log likelihood for all samples
            log_prior = self.class_log_prior_[idx]
            # Compute log probability under Gaussian for each sample: shape (n_samples,)
            # Vectorized: for all samples X, sum over features
            # Equivalent to [self._gaussian_log_prob(idx, x) for x in X]
            mean = self.theta_[idx]
            var = self.sigma_[idx]
            log_coeff = -0.5 * np.sum(np.log(2.0 * np.pi * var))
            # Compute sum((x - mean)^2 / var) for each sample:
            diff = X - mean  # shape (n_samples, n_features)
            log_exp = -0.5 * np.sum((diff ** 2) / var, axis=1)
            jll[:, idx] = log_prior + log_coeff + log_exp

        # Choose the class with highest joint log-likelihood for each sample
        class_indices = np.argmax(jll, axis=1)
        return self.classes_[class_indices]


# Example usage:
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # --- MultinomialNB Example ---
    # For demonstration, convert continuous features to counts by rounding (not typical for Iris,
    # but just to illustrate MultinomialNB usage).
    X_train_counts = np.round(X_train).astype(int)
    X_test_counts = np.round(X_test).astype(int)

    mnb = MultinomialNB(alpha=1.0)
    mnb.fit(X_train_counts, y_train)
    y_pred_mnb = mnb.predict(X_test_counts)
    print("MultinomialNB accuracy (rounded counts):", accuracy_score(y_test, y_pred_mnb))

    # --- GaussianNB Example ---
    gnb = GaussianNB(var_smoothing=1e-9)
    gnb.fit(X_train, y_train)
    y_pred_gnb = gnb.predict(X_test)
    print("GaussianNB accuracy:", accuracy_score(y_test, y_pred_gnb))
