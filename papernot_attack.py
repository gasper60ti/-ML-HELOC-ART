import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from art.attacks.evasion import DecisionTreeAttack
from art.estimators.classification import SklearnClassifier

# Load the preprocessed data
X_train = np.load('X_train.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_test = np.load('y_test.npy')

# Load the trained Random Forest model from `02_random_forest.py`
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)  # Make sure the Random Forest is re-trained with the same settings

# Use a single tree from the Random Forest for the attack
tree_model = rf.estimators_[0]
classifier = SklearnClassifier(model=tree_model)

# Generate adversarial examples
attack = DecisionTreeAttack(classifier=classifier, offset=0.001, verbose=True)
X_test_adv = attack.generate(X_test)

# Evaluate the Random Forest on the adversarial examples
y_pred_adv_rf = rf.predict(X_test_adv)
accuracy_adv_rf = accuracy_score(y_test, y_pred_adv_rf)
print(f"Random Forest Accuracy on Adversarial Examples: {accuracy_adv_rf:.4f}")
