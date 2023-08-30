import unittest
from app import app
import json

class TestApp(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_predict(self):
        response = self.app.post('/predict', json={
            "Air_temperature": 1,
            "Process_temperature": 2,
            "Rotational_speed": 3,
            "Torque": 4,
            "Tool_wear": 5
        })
        
        data = json.loads(response.get_data())

        self.assertIn('prediction', data)
        self.assertIn(data['prediction'], ['Failure', 'No Failure'])

if __name__ == '__main__':
    unittest.main()
