import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset (replace with actual dataset path)
file_path = "PPG_Dataset.csv"  
df = pd.read_csv(file_path)

# Select only Heart Rate (HR) & SpO₂ as features
X = df[['Heart_Rate', 'SpO2']].values  
y = df['Label'].values  # Label: 0 = Normal, 1 = Abnormal

# Normalize input features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Reshape for LSTM input (samples, timesteps, features)
X = X.reshape(X.shape[0], 1, 2)  # 1 timestep, 2 features (HR & SpO₂)

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(1, 2)),  
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("heart_monitor_lstm.h5")
np.save("scaler.npy", scaler.mean_)  # Save scaler for real-time data

print("✅ Model training complete!")
