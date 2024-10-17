import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load the preprocessed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Train the Random Forest model
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate the model
y_pred_rf = rf.predict_proba(X_test)[:, 1]
print(y_pred_rf)
auc_roc_rf = roc_auc_score(y_test, y_pred_rf)
print(f"Random Forest AUC-ROC Score: {auc_roc_rf:.4f}")
