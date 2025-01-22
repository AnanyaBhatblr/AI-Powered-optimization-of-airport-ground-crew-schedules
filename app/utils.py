# app/utils.py
import pandas as pd

def format_task_data(data):
    df = pd.DataFrame(data)
    df['Task Duration'] = (pd.to_datetime(df['End Time']) - pd.to_datetime(df['Start Time'])).dt.total_seconds() / 60
    return df

def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {file_path}. Exception: {e}")
        return None
