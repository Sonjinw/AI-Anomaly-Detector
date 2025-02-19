import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt

def generate_data():
    # Generate synthetic data with inliers and outliers
    X_inliers = 0.3 * np.random.randn(100, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X_inliers, X_outliers]
    return X

def detect_anomalies(X):
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)
    y_pred = clf.predict(X)
    return y_pred

if __name__ == "__main__":
    X = generate_data()
    predictions = detect_anomalies(X)
    plt.scatter(X[:, 0], X[:, 1], c=predictions, cmap='coolwarm')
    plt.title("Anomaly Detection using Isolation Forest")
    plt.show()
