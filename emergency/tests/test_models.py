# tests/test_models.py
import unittest
import pandas as pd
import pickle

class TestModels(unittest.TestCase):
    def setUp(self):
        with open('../models/kmeans_model.pkl', 'rb') as f:
            self.kmeans_model = pickle.load(f)

        with open('../models/arima_model.pkl', 'rb') as f:
            self.arima_model = pickle.load(f)

    def test_kmeans_model(self):
        data = pd.DataFrame({'Task Type': [1, 2], 'Task Duration': [30, 60]})
        predictions = self.kmeans_model.predict(data)
        self.assertTrue(len(predictions) > 0)

    def test_arima_model(self):
        forecast = self.arima_model.forecast(steps=5)
        self.assertEqual(len(forecast), 5)

if __name__ == "__main__":
    unittest.main()
