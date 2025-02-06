# app/api.py
import pandas as pd
import pickle
from flask import Flask, request, jsonify
from .incident_manager import analyze_incident_report, optimize_crew_routing

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
    # Forecast for the next 10 periods
    forecast = arima_model.forecast(steps=10)
    return jsonify({'forecast': forecast.tolist()})


@app.route('/report_incident', methods=['POST'])
def report_incident():
    """
    Example endpoint to classify new incident text, then dynamically reallocate crew.
    Request JSON: {'IncidentID': 'INC123', 'ReportText': 'Spilled fuel on runway', ...}
    """
    data = request.json
    report_text = data.get("ReportText", "")
    severity = analyze_incident_report(report_text)
    # ...retrieve relevant incident & crew data frames...
    incidents_df = pd.DataFrame(
        [{"IncidentID": data["IncidentID"], "Severity": severity}])
    crew_df = pd.DataFrame([{"Crew ID": "CRW0001"}])  # placeholder
    result = optimize_crew_routing(incidents_df, crew_df)
    return jsonify(result.to_dict(orient='records'))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
