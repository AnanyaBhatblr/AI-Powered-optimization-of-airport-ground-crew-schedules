# Crew Scheduler Project

## Overview
An AI-powered solution to optimize airport ground crew schedules.

## Features
- Task clustering using K-Means.
- Forecasting crew demand with ARIMA.
- Real-time dynamic allocation with Reinforcement Learning.
- Optimization of schedules using MILP.

## How to Run
1. Install dependencies: `pip install -r requirements.txt`.
2. Run the app: `streamlit run app/app.py`.
3. For Docker, build the image and run:
   ```bash
   docker build -t crew-scheduler .
   docker run -p 8501:8501 crew-scheduler
