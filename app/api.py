# app/api.py
import pandas as pd
import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load models
with open('../models/kmeans_model.pkl', 'rb') as f:
    kmeans_model = pickle.load(f)

with open('../models/arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

@app.route('/cluster_tasks', methods=['POST'])
def cluster_tasks():
    data = request.json
    df = pd.DataFrame(data)
    task_features = pd.get_dummies(df[['Task Type', 'Task Duration']])
    clusters = kmeans_model.predict(task_features)
    df['Cluster'] = clusters
    return jsonify(df.to_dict(orient='records'))

@app.route('/forecast_demand', methods=['GET'])
def forecast_demand():
    forecast = arima_model.forecast(steps=10)  # Forecast for the next 10 periods
    return jsonify({'forecast': forecast.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)