import unittest
import json
import pandas as pd
import joblib
import numpy as np
from app import app

class TestHousePricePrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the model for testing
        cls.model = joblib.load('model.pkl')
        cls.app = app.test_client()
        cls.app.testing = True

        # Example of expected columns (you might need to adjust this list)
        cls.expected_columns = [
            'area', 'bedrooms', 'bathrooms', 'stories', 'parking',
            'mainroad_yes', 'guestroom_yes', 'basement_yes',
            'hotwaterheating_yes', 'airconditioning_yes', 'prefarea_yes',
            'furnishingstatus_semi-furnished', 'furnishingstatus_furnished'
        ]

    def test_model_prediction(self):
        test_data = pd.DataFrame({
            'area': [2000],
            'bedrooms': [3],
            'bathrooms': [2],
            'stories': [2],
            'parking': [2],
            'mainroad_yes': [1],
            'guestroom_yes': [0],
            'basement_yes': [0],
            'hotwaterheating_yes': [0],
            'airconditioning_yes': [1],
            'prefarea_yes': [1],
            'furnishingstatus_furnished': [0],
            'furnishingstatus_unfurnished': [1]
        })
        
        prediction = self.model.predict(test_data)
        self.assertIsInstance(prediction, np.ndarray)  # Check if prediction is an ndarray
        self.assertEqual(prediction.shape, (1,))  # Ensure it has shape (1,)
        self.assertIsInstance(prediction[0], (float, int))  # Check if first element is float or int

    def test_api_prediction(self):
        # Test the /predict API endpoint
        response = self.app.post('/predict', 
                                 data=json.dumps({
                                     'area': 1000,
                                     'bedrooms': 3,
                                     'bathrooms': 2,
                                     'stories': 2,
                                     'mainroad': 'yes',
                                     'guestroom': 'no',
                                     'basement': 'yes',
                                     'hotwaterheating': 'no',
                                     'airconditioning': 'yes',
                                     'parking': 2,
                                     'prefarea': 'yes',
                                     'furnishingstatus': 'semi-furnished'
                                 }),
                                 content_type='application/json')
        self.assertEqual(response.status_code, 200)
        response_json = json.loads(response.data)
        self.assertIn('prediction', response_json)
        self.assertIsInstance(response_json['prediction'], (float, int))
        self.assertGreater(response_json['prediction'], 0)

if __name__ == '__main__':
    unittest.main()
