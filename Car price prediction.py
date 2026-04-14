#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 1: Load Dataset
file_path = 'car data.csv'
car_data = pd.read_csv(file_path)

# Step 2: Feature Engineering
car_data['Car_Age'] = 2025 - car_data['Year']
car_data.drop(['Year', 'Car_Name'], axis=1, inplace=True)

# Step 3: One-Hot Encoding for Categorical Variables
car_data = pd.get_dummies(car_data, drop_first=True)

# Step 4: Prepare Features and Target
X = car_data.drop('Selling_Price', axis=1)
y = car_data['Selling_Price']

# Step 5: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Step 8: Predict and Evaluate
y_pred = model.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error (MAE): {mae:.2f}')
print(f'Mean Squared Error (MSE): {mse:.2f}')
print(f'R² Score: {r2:.2f}')


# In[ ]:




