import numpy as np
import gzip
import numpy as np
from scipy.stats import mode
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt



def load_gzipped_data(file_path):
    """
    Load and return the labels and features from a gzipped file containing digit data.
    
    Parameters:
    - file_path: Path to the gzipped file.
    
    Returns:
    - labels: Numpy array of digit labels.
    - features: Numpy array of digit image features, normalized to [0, 1].
    """
    with gzip.open(file_path, 'rt') as file:  # Open the gzipped file in text mode
        # Initialize lists to store labels and features
        labels = []
        features = []
        
        # Process each line in the file
        for line in file:
            
            tokens = line.strip().split()
            labels.append(int(float(tokens[0])))  # First token is the label
            features.append([int(float((value))) for value in tokens[1:]])  # Remaining tokens are features
            
        # Convert lists to numpy arrays and normalize features
        labels = np.array(labels)
        features = np.array(features) / 255.0  # Normalize to [0, 1]
        
    return labels, features

# Load training and testing data
train_labels, train_features = load_gzipped_data('/Users/margokim/Downloads/zip.train.gz')
test_labels, test_features = load_gzipped_data('/Users/margokim/Downloads/zip.test.gz')

def nadaraya_watson_knn(X_train, y_train, X_test, k=5):
    """
    k-NN classification using the Nadaraya-Watson kernel method.
    
    Parameters:
    - X_train: Training features
    - y_train: Training labels
    - X_test: Testing features
    - k: Number of neighbors
    
    Returns:
    - predictions: Predicted labels for X_test
    """
    distances = pairwise_distances(X_test, X_train, metric='euclidean')
    knn_indices = np.argsort(distances, axis=1)[:, :k]
    knn_labels = y_train[knn_indices]
    weights = 1 / (distances[np.arange(distances.shape[0])[:, None], knn_indices] + 1e-5)
    weighted_labels = np.array([np.bincount(labels, weights=weight, minlength=10) for labels, weight in zip(knn_labels, weights)])
    predictions = np.argmax(weighted_labels, axis=1)
    return predictions

def calculate_accuracy(y_true, y_pred):
    
    return np.mean(y_true == y_pred)

predictions = nadaraya_watson_knn(train_features, train_labels, test_features, k=5)
accuracy = calculate_accuracy(test_labels, predictions)

print(f"Model accuracy: {accuracy:.4f}")



def zero_one_loss(y_true, y_pred):
    """
    Compute the zero-one loss function - a simple error count.
    
    Parameters:
    - y_true: Numpy array of true labels
    - y_pred: Numpy array of predicted labels
    
    Returns:
    - loss: Zero-one loss (number of misclassifications)
    """
    return np.sum(y_true != y_pred)

def k_fold_cross_validation_knn(X, y, k_range=range(1, 21), folds=5):
    """
    Estimate the average test error for each 'k' in k-NN using 5-fold cross-validation.
    
    Parameters:
    - X: Features in the dataset
    - y: Labels in the dataset
    - k_range: Range of 'k' values to test
    - folds: Number of folds for cross-validation
    
    Returns:
    - average_errors: Dictionary mapping 'k' to its average zero-one loss across folds
    """
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    average_errors = {k: [] for k in k_range}
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        for k in k_range:
            predictions = nadaraya_watson_knn(X_train, y_train, X_test, k=k)
            loss = zero_one_loss(y_test, predictions)
            average_errors[k].append(loss)
    
    # Compute the average zero-one loss for each 'k'
    for k in average_errors:
        average_errors[k] = np.mean(average_errors[k]) / len(y_test)  # Normalize by test set size
    
    return average_errors


k_range = range(1, 21)
average_test_errors = k_fold_cross_validation_knn(train_features, train_labels, k_range=k_range, folds=5)

# Print average test errors for each 'k'
for k, error in average_test_errors.items():
    print(f"k={k}: Average Test Error={error:.4f}")


# Assuming average_test_errors is obtained from the modified k_fold_cross_validation_knn function
average_test_errors = k_fold_cross_validation_knn(train_features, train_labels, k_range=range(1, 21), folds=5)

# Extracting values for plotting
ks = list(average_test_errors.keys())
average_errors = [average_test_errors[k]['average_error'] for k in ks]
std_errors = [average_test_errors[k]['std_error'] for k in ks]

# Plotting
plt.figure(figsize=(10, 6))
plt.errorbar(ks, average_errors, yerr=std_errors, fmt='-o', capsize=5, capthick=2, ecolor='r', marker='s', markersize=5, linestyle='-', linewidth=1, label='Average Test Error')
plt.title('Average Test Error vs. Number of Neighbors (k) with Standard Error')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Average Test Error')
plt.xticks(ks)
plt.legend()
plt.grid(True)
plt.show()


