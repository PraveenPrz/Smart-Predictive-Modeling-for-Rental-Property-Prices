import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib  # Use joblib for serialization

# Load the data
df = pd.read_excel('House_Rent_Train.xlsx')
df = df.dropna()

# Separate features and target variable
X = df.drop('rent', axis=1)
y = df['rent']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps for numeric and categorical features
numeric_features = ['latitude', 'longitude', 'property_size', 'property_age', 'bathroom', 'floor', 'total_floor', 'balconies']
categorical_features = ['gym', 'lift', 'swimming_pool', 'negotiable', 'cup_board']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
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

# Train the model
model = RandomForestRegressor()
model.fit(X_train_preprocessed, y_train)

# Save the preprocessor and model to disk
joblib.dump(preprocessor, 'preprocessor.joblib')
joblib.dump(model, 'random_forest_model.joblib')