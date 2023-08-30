import unittest
from app import app  # Adjust the import path as needed

class TestAPI(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_predict_endpoint(self):
        response = self.app.post('/predict', json={
            'Air_temperature': 298.1,
            'Process_temperature': 308.6,
            'Rotational_speed': 1551,
            'Torque': 42.8,
            'Tool_wear': 0
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn('prediction', response.get_json())
