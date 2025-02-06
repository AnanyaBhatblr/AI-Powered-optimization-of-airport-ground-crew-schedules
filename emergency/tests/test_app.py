# tests/test_app.py
import unittest
from app.utils import format_task_data

class TestApp(unittest.TestCase):
    def test_format_task_data(self):
        data = [
            {"Task ID": 1, "Task Type": "Refueling", "Start Time": "2023-01-01 08:00:00", "End Time": "2023-01-01 08:30:00"},
            {"Task ID": 2, "Task Type": "Cleaning", "Start Time": "2023-01-01 09:00:00", "End Time": "2023-01-01 09:45:00"}
        ]
        formatted_data = format_task_data(data)
        self.assertEqual(formatted_data.iloc[0]['Task Duration'], 30)
        self.assertEqual(formatted_data.iloc[1]['Task Duration'], 45)

if __name__ == "__main__":
    unittest.main()
