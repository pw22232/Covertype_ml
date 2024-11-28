import numpy as np
from itertools import combinations
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split

random_seed = 42  # Fixed random seed
np.random.seed(random_seed)
n_samples = 10000
K = 7  # Set the number of clusters to 7

def load_data():
    data = fetch_covtype()
    data_X = data.data
    data_y = data.target
    X, _, y, _ = train_test_split(data_X, data_y, train_size=n_samples,
                                  random_state=random_seed)  # randomly chosen subset of 10,000 datapoints
    return X, y

def kmeans_method(X):
    kmeans = KMeans(n_clusters=K, random_state=random_seed)
    kmeans_labels = kmeans.fit_predict(X)
    return kmeans_labels

def gmm_method(X):
    gmm = GaussianMixture(n_components=K, random_state=random_seed)
    gmm_labels = gmm.fit_predict(X)
    return gmm_labels

def random_baseline():
    return np.random.randint(0, K, size=n_samples)

total_pairs = 0
def calculate_errors_and_accuracy(y, labels):
    global total_pairs
    errors = 0
    correct = 0
    label_pairs = defaultdict(list)

    # Group indices by true class label
    for index, label in enumerate(y):
        label_pairs[label].append(index)

    # Check each pair of points within the same class
    for indices in label_pairs.values():
        for i, j in combinations(indices, 2):
            if labels[i] != labels[j]:
                errors += 1
            else:
                correct += 1

    total_pairs = errors + correct
    accuracy = correct / total_pairs

    return errors, accuracy


if __name__ == "__main__":
    # Code Task 1
    X, y = load_data()
    # Code Task 2
    kmeans_labels = kmeans_method(X)
    # Code Task 3
    gmm_labels = gmm_method(X)
    # Code Task 4
    random_labels = random_baseline()

    # Code Task 5
    kmeans_errors, kmeans_accuracy = calculate_errors_and_accuracy(y, kmeans_labels)
    gmm_errors, gmm_accuracy = calculate_errors_and_accuracy(y, gmm_labels)
    random_errors, random_accuracy = calculate_errors_and_accuracy(y, random_labels)

    print("Total pairs with the same class label:", total_pairs)
    print("K-means Errors:", kmeans_errors, ", Accuracy:", f"{kmeans_accuracy * 100: .2f}%")
    print("GMM Errors:", gmm_errors, ", Accuracy:", f"{gmm_accuracy * 100: .2f}%")
    print("Random baseline Errors:", random_errors, ", Accuracy:", f"{random_accuracy * 100: .2f}%")
