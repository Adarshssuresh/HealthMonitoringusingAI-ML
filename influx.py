import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
file_path = "/mnt/data/PPG_Dataset.csv"
df = pd.read_csv(file_path)

# Select only PPG-related fields (HR, SpO2) as features
X = df[['Heart_Rate', 'SpO2']].values  # Only HR & SpO2
y = df['Label'].values  # Binary classification (0 = Normal, 1 = Abnormal)

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape X to fit LSTM input shape (samples, timesteps=1, features)
X = X.reshape(X.shape[0], 1, X.shape[1])  # (samples, 1 time step, 2 features)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the LSTM model
model = Sequential([
    LSTM(64, input_shape=(1, X.shape[2])),  # Single timestep
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("heart_monitor_lstm_no_timesteps.h5")
print("✅ Model training complete without time steps!")
