# scripts/data_preprocessing.py
import pandas as pd
from datetime import datetime


def preprocess_tasks(file_path):
    tasks = pd.read_csv(file_path)
    # Convert timestamps to datetime
    tasks['Start_Time'] = pd.to_datetime(tasks['Start_Time'])
    tasks['End_Time'] = pd.to_datetime(tasks['End_Time'])
    # Calculate duration in minutes properly
    tasks['Task_Duration'] = (
        tasks['End_Time'] - tasks['Start_Time']).dt.total_seconds() / 60
    return tasks


def preprocess_crew(file_path):
    crew = pd.read_csv(file_path)
    return crew


def preprocess_disruptions(file_path):
    disruptions = pd.read_csv(file_path)
    return disruptions


if __name__ == "__main__":
    tasks = preprocess_tasks("../data/tasks.csv")
    crew = preprocess_crew("../data/crew.csv")
    disruptions = preprocess_disruptions("../data/disruptions.csv")

    tasks.to_csv("../data/processed_tasks.csv", index=False)
    crew.to_csv("../data/processed_crew.csv", index=False)
    disruptions.to_csv("../data/processed_disruptions.csv", index=False)

    print("Preprocessing complete!")
