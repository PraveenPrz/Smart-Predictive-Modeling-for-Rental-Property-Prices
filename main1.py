import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

# Load the data
df = pd.read_excel('House_Rent_Train.xlsx')
df.isnull().sum()
df=df.dropna()
df.isnull().sum()


# Separate features and target variable
X = df.drop('rent', axis=1)
y = df['rent']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numeric and categorical features
numeric_features = ['latitude', 'longitude', 'property_size', 'property_age', 'bathroom', 'floor', 'total_floor', 'balconies']
categorical_features = ['gym', 'lift', 'swimming_pool', 'negotiable', 'cup_board']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # You can use other strategies like 'median' or 'most_frequent'
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine the transformers using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the training data
X_train_preprocessed = preprocessor.fit_transform(X_train)

# Preprocess the testing data
X_test_preprocessed = preprocessor.transform(X_test)

# Get the feature names after one-hot encoding
feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features)

# Combine numeric and categorical feature names
all_feature_names = numeric_features + list(feature_names)

# Feature selection can be performed based on model importance scores
# You may use methods like RandomForestRegressor to get feature importances
model = RandomForestRegressor()
model.fit(X_train_preprocessed, y_train)

# Display feature importances
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index=all_feature_names,
                                   columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances)

from sklearn.ensemble import RandomForestRegressor  # You can choose another algorithm based on your preference

# Initialize the model
model = RandomForestRegressor()

# Train the model
model.fit(X_train_preprocessed, y_train)
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Predict on the test set
y_pred = model.predict(X_test_preprocessed)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'Mean Absolute Error: {mae}')
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Initialize the model
model = RandomForestRegressor()

# Perform Grid Search Cross Validation
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_preprocessed, y_train)

# Get the best parameters
best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

# Train the model with the best parameters
best_model = grid_search.best_estimator_
best_model.fit(X_train_preprocessed, y_train)

# Evaluate the best model
y_pred_best = best_model.predict(X_test_preprocessed)
mse_best = mean_squared_error(y_test, y_pred_best)
mae_best = mean_absolute_error(y_test, y_pred_best)

print(f'Best Model Mean Squared Error: {mse_best}')
print(f'Best Model Mean Absolute Error: {mae_best}')
# If your selected model supports feature importances, you can visualize them
feature_importances_best = pd.DataFrame(best_model.feature_importances_,
                                         index=all_feature_names,
                                         columns=['importance']).sort_values('importance', ascending=False)
print(feature_importances_best)
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)


# Define a route to predict rent
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame(data)

    # Preprocess the input data
    input_data_preprocessed = preprocessor.transform(input_data)

    # Make predictions using the best model
    prediction = best_model.predict(input_data_preprocessed)

    return jsonify({'prediction': prediction.tolist()})


# Run the app
if __name__ == '__main__':
    app.run(port=5000, debug=True)