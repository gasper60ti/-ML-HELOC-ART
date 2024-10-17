import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load and preprocess the data
url = './heloc_dataset.csv'
data = pd.read_csv(url)

# print(data.head())

# Fill missing values in 'RiskPerformance' with 'Bad'
data['RiskPerformance'].fillna('Bad', inplace=True)

# Handling missing values for numeric columns only
numeric_columns = data.select_dtypes(include=[np.number]).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Convert categorical to numerical
data['RiskPerformance'] = data['RiskPerformance'].apply(lambda x: 1 if x == 'Good' else 0)

# Separate features and target
X = data.drop('RiskPerformance', axis=1)
y = data['RiskPerformance']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print(X_train[0], y_train.iloc[0])
# Save preprocessed data to disk
np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)
