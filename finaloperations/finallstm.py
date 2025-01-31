import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# for bengaluru T2
df = pd.read_csv("../dataset/BT1/output2BT1.csv")
# Prepare the data
categorical_cols = ['task_ID', 'Gate_number', 'Floor_No', 'shift_no']
df[categorical_cols] = df[categorical_cols].astype('category')

sequences = []
target = []
scalers = {}  # Store scalers for each group
for name, group in df.groupby(categorical_cols, observed=False):  # Pass observed=False
    demand_values = group['crew_demand'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_demand = scaler.fit_transform(demand_values)
    scalers[name] = scaler  # Store the scaler for this group
    scaled_demand = scaled_demand.flatten()

    seq_length = 10
    for i in range(len(scaled_demand) - seq_length):
        sequences.append((name, scaled_demand[i:i + seq_length]))  # Store the name of the group with the sequence
        target.append(scaled_demand[i + seq_length])

# Prepare input and output data
X = np.array([seq[1] for seq in sequences])
y = np.array(target)
groups = [seq[0] for seq in sequences]

# Split the data into train and test sets
X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
    X, y, groups, test_size=0.2, random_state=42, stratify=groups
)

# Reshape data for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = keras.Sequential([
    keras.Input(shape=(X_train.shape[1], X_train.shape[2])),  # Define input shape
    keras.layers.LSTM(64, activation='tanh', return_sequences=True),
    keras.layers.LSTM(32, activation='tanh'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=1)

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Mean Absolute Error on Scaled Test Set: {mae}")
predictions_scaled = model.predict(X_test)

# Transform predictions back to original scale
predictions = []
y_test_original = []
groups_used = []

for i, group in enumerate(groups_test):
    scaler = scalers[group]
    prediction_scaled = predictions_scaled[i].reshape(-1, 1)
    prediction = scaler.inverse_transform(prediction_scaled).flatten()
    prediction_adjusted = np.where(prediction < 0, np.ceil(prediction), np.floor(prediction))  # Adjust prediction based on sign
    actual = scaler.inverse_transform(y_test[i].reshape(-1, 1)).flatten()
    predictions.append(prediction_adjusted[0])
    y_test_original.append(actual[0])
    groups_used.append(group)

# Calculate evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_original, predictions))
print(f"Root Mean Squared Error on Test Set: {rmse}")
mae = mean_absolute_error(y_test_original, predictions)
print(f"Mean Absolute Error on Test Set: {mae}")

# Save predictions to a CSV file
predictions_df = pd.DataFrame({
    'Task_ID': [group[0] for group in groups_used],
    'Gate_number': [group[1] for group in groups_used],
    'Floor_No': [group[2] for group in groups_used],
    'Shift_no': [group[3] for group in groups_used],
    'Actual': y_test_original,
    'Predicted': predictions
})

predictions_df.to_csv("../dataset/BT1/predBT1.csv", index=False)
print(f"Predictions saved to ../data/predictions_output.csv")

# Example of a single prediction
last_group = groups_test[-1]
last_scaler = scalers[last_group]
last_sequence = X_test[-1]
last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))

predicted_value_scaled = model.predict(last_sequence)
predicted_value = last_scaler.inverse_transform(predicted_value_scaled).flatten()
predicted_value_adjusted = np.where(predicted_value < 0, np.ceil(predicted_value), np.floor(predicted_value))
