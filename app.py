from flask import Flask, request, jsonify
import pandas as pd
import joblib  # For scikit-learn versions < 0.23
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)

# Load the preprocessor and model from disk
preprocessor = joblib.load('preprocessor.joblib')
model = joblib.load('random_forest_model.joblib')

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        data = request.json

        # Preprocess input data
        input_data = pd.DataFrame(data, index=[0])  # Convert JSON to DataFrame
        input_preprocessed = preprocessor.transform(input_data)

        # Make prediction
        prediction = model.predict(input_preprocessed)

        # Replace the following line with the actual result based on your model
        result = {'rent': prediction[0]}
        print(result)


        return jsonify(result)
    elif request.method == 'GET':
        return jsonify({'message': 'GET request received'})

if __name__ == '__main__':
    app.run(debug=True)