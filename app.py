import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

# Load the model and columns
model = joblib.load('model.pkl')
columns = joblib.load('columns.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Convert categorical data to the same format as the model
    def binary_map(val):
        return 1 if val == 'yes' else 0
    
    def furnishing_status_map(val):
        if val == 'furnished':
            return 2
        elif val == 'semi-furnished':
            return 1
        else:  # 'unfurnished'
            return 0

    # Create a feature list based on columns from training
    features = {
        'area': float(data.get('area', 0)),
        'bedrooms': int(data.get('bedrooms', 0)),
        'bathrooms': int(data.get('bathrooms', 0)),
        'stories': int(data.get('stories', 0)),
        'parking': int(data.get('parking', 0)),
        'mainroad_yes': binary_map(data.get('mainroad', 'no')),
        'guestroom_yes': binary_map(data.get('guestroom', 'no')),
        'basement_yes': binary_map(data.get('basement', 'no')),
        'hotwaterheating_yes': binary_map(data.get('hotwaterheating', 'no')),
        'airconditioning_yes': binary_map(data.get('airconditioning', 'no')),
        'prefarea_yes': binary_map(data.get('prefarea', 'no')),
        'furnishingstatus_furnished': furnishing_status_map(data.get('furnishingstatus', 'unfurnished')) == 2,
        'furnishingstatus_semi-furnished': furnishing_status_map(data.get('furnishingstatus', 'unfurnished')) == 1,
        'furnishingstatus_unfurnished': furnishing_status_map(data.get('furnishingstatus', 'unfurnished')) == 0
    }

    # Make sure all columns are included in the feature array
    feature_values = np.array([[features.get(col, 0) for col in columns]])

    # Make the prediction using the model
    prediction = model.predict(feature_values)[0]
    
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
