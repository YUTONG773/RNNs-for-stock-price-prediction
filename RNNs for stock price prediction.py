#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load dataset
data = pd.read_csv('Google_Stock_Price_Test.csv')

# Display dataset info and first rows
print(data.info())
print(data.head())

# Convert Volume column to float
data['Volume'] = data['Volume'].str.replace(',', '').astype(float)

# Convert Date column to datetime
data['Date'] = pd.to_datetime(data['Date'])

# Confirm data type changes
print(data.info())
print(data.head())

# Check if Close > High
anomalies_close_high = data[data['Close'] > data['High']]
print("Close > High anomalies:")
print(anomalies_close_high)

# Check if Open < Low
anomalies_open_low = data[data['Open'] < data['Low']]
print("Open < Low anomalies:")
print(anomalies_open_low)

# Save cleaned dataset
data.to_csv('cleaned_dataset.csv', index=False)
print("Data cleaning completed and saved!")


# In[2]:


# Load cleaned dataset
data = pd.read_csv('cleaned_dataset.csv')

# Define features and target for prediction
N = 5  # Use past 5 days as input
features = ['Open', 'High', 'Low', 'Close', 'Volume']
target = 'Close'

# Create input features and target variable
data_prepared = []
for i in range(N, len(data)):
    past_features = data[features].iloc[i-N:i].values.flatten()  # Last N days data
    target_value = data[target].iloc[i]  # Current day's Close
    data_prepared.append((past_features, target_value))

# Convert to DataFrame
data_prepared = pd.DataFrame(data_prepared, columns=['Features', 'Target'])

# Split into training and testing datasets
train_size = int(len(data_prepared) * 0.8)  # 80% for training
train_data = data_prepared.iloc[:train_size]
test_data = data_prepared.iloc[train_size:]

print("Training data size:", train_data.shape)
print("Testing data size:", test_data.shape)


# In[3]:


from sklearn.model_selection import train_test_split

# Convert Features and Target to NumPy arrays
X_train = np.array(train_data['Features'].tolist())
y_train = np.array(train_data['Target'].tolist())
X_test = np.array(test_data['Features'].tolist())
y_test = np.array(test_data['Target'].tolist())

# Display shapes of training and testing data
print("Training Features shape:", X_train.shape)
print("Training Target shape:", y_train.shape)
print("Testing Features shape:", X_test.shape)
print("Testing Target shape:", y_test.shape)


# In[4]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout, Input

# Reshape features to match RNN input format (samples, timesteps, features)
X_train = X_train.reshape(X_train.shape[0], 5, 5)  # 5 timesteps, 5 features
X_test = X_test.reshape(X_test.shape[0], 5, 5)

# Function to build different RNN models
def build_rnn_model(rnn_type='LSTM'):
    model = Sequential()
    model.add(Input(shape=(5, 5)))  # Specify input shape

    if rnn_type == 'VanillaRNN':
        model.add(SimpleRNN(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(SimpleRNN(50, return_sequences=False))
    elif rnn_type == 'GRU':
        model.add(GRU(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(GRU(50, return_sequences=False))
    elif rnn_type == 'LSTM':
        model.add(LSTM(50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))

    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer for single target prediction
    return model


# In[5]:


# Set model type to Vanilla RNN
model_type = 'VanillaRNN'

# Build and compile model
model = build_rnn_model(rnn_type=model_type)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Type: {model_type}")
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)

# Set model type to GRU
model_type = 'GRU'

# Build and compile model
model = build_rnn_model(rnn_type=model_type)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Type: {model_type}")
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)

# Set model type to LSTM
model_type = 'LSTM'

# Build and compile model
model = build_rnn_model(rnn_type=model_type)
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
print(f"Model Type: {model_type}")
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)


# In[6]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout, Input
import matplotlib.pyplot as plt

# Build RNN model with specified parameters
def build_rnn_model(rnn_type='LSTM', units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(Input(shape=(5, 5)))  # Input shape: 5 timesteps, 5 features
    if rnn_type == 'VanillaRNN':
        model.add(SimpleRNN(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(SimpleRNN(units, return_sequences=False))
    elif rnn_type == 'GRU':
        model.add(GRU(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(GRU(units, return_sequences=False))
    elif rnn_type == 'LSTM':
        model.add(LSTM(units, return_sequences=True))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Output layer
    return model

# Train and evaluate the model
def train_and_evaluate_optimized(model_type, units=64, dropout_rate=0.3, epochs=100, batch_size=8):
    model = build_rnn_model(rnn_type=model_type, units=units, dropout_rate=dropout_rate)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, 
                        validation_data=(X_test, y_test), verbose=0)
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    return history, test_loss, test_mae

# Define model types to test
model_types = ['VanillaRNN', 'GRU', 'LSTM']

# Visualize training and validation loss
def plot_history(histories, model_names):
    plt.figure(figsize=(12, 6))
    for history, name in zip(histories, model_names):
        plt.plot(history.history['loss'], label=f'{name} - Training Loss')
        plt.plot(history.history['val_loss'], label=f'{name} - Validation Loss')
    plt.title('Model Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

# Optimize models and store results
optimized_results = []
optimized_histories = []

for model_type in model_types:  # VanillaRNN, GRU, LSTM
    print(f"Optimizing {model_type}...")
    history, test_loss, test_mae = train_and_evaluate_optimized(
        model_type=model_type, units=64, dropout_rate=0.3, epochs=100, batch_size=8)
    optimized_histories.append(history)
    optimized_results.append((model_type, test_loss, test_mae))

# Print performance of optimized models
print("\nOptimized Model Performance:")
for model_type, test_loss, test_mae in optimized_results:
    print(f"{model_type}: Test Loss (MSE) = {test_loss}, Test MAE = {test_mae}")

# Plot training curves
plot_history(optimized_histories, model_types)


# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load datasets
train_data = pd.read_csv('Google_Stock_Price_Test.csv')
test_data = pd.read_csv('Google_Stock_Price_Test.csv')

# Preprocess the data
def preprocess_data(data, features):
    data = data[features]
    data = data.replace(',', '', regex=True).astype(float)
    return data

features = ['Open', 'High', 'Low', 'Close', 'Volume']
train_data_cleaned = preprocess_data(train_data, features)
test_data_cleaned = preprocess_data(test_data, features)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data_cleaned)
test_scaled = scaler.transform(test_data_cleaned)

# Check the data shape
print("Train scaled shape:", train_scaled.shape)
print("Test scaled shape:", test_scaled.shape)

# Create time-series data
def create_time_series(data, look_back=30):
    X, y = [], []
    for i in range(look_back, len(data) - 1):
        X.append(data[i - look_back:i])
        y.append(data[i, 3])  # Target is the 'Close' price
    X = np.array(X)
    if len(X.shape) == 2:  # 如果是二维的，添加特征维度
        X = np.expand_dims(X, axis=-1)
    print(f"Generated {len(X)} samples.")
    return X, np.array(y)

# Adjust look_back dynamically
look_back = min(5, train_scaled.shape[0] - 1)  # 调整 look_back 为更小的值
print("Using look_back:", look_back)

# Generate time-series data
X_train, y_train = create_time_series(train_scaled, look_back)
if X_train.size == 0 or y_train.size == 0:
    raise ValueError("Not enough data to generate time-series. Try reducing look_back or increasing dataset size.")

print("X_train shape after time-series generation:", X_train.shape)
print("y_train shape after time-series generation:", y_train.shape)

# Adjust test dataset for insufficient data
def create_test_series(train_data, test_data, look_back):
    combined_data = np.vstack([train_data[-look_back:], test_data])
    return create_time_series(combined_data, look_back)

X_test, y_test = create_test_series(train_scaled, test_scaled, look_back)
if X_test.size == 0 or y_test.size == 0:
    raise ValueError("Not enough data to generate test series. Try reducing look_back or increasing dataset size.")

print("X_test shape after time-series generation:", X_test.shape)
print("y_test shape after time-series generation:", y_test.shape)

# Input shape for the model
input_shape = (look_back, X_train.shape[2])

# Define LSTM model
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build and train the LSTM model
lstm_model = build_lstm_model(input_shape)
lstm_history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# Evaluate the LSTM model
def evaluate_model(model, X, y_true):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

lstm_mae, lstm_rmse = evaluate_model(lstm_model, X_test, y_test)

# Print Results
print(f"LSTM - MAE: {lstm_mae}, RMSE: {lstm_rmse}")

# Visualizations
# Training and Validation Loss
def plot_training_history(history, title):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{title} - Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(lstm_history, "LSTM")

# Predictions vs True Values
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true, label='True Values', marker='o')
    plt.plot(y_pred, label='Predicted Values', marker='x')
    plt.title(f'{title} - Predictions vs True Values')
    plt.xlabel('Time Steps')
    plt.ylabel('Stock Price (Normalized)')
    plt.legend()
    plt.show()

lstm_predictions = lstm_model.predict(X_test)
plot_predictions(y_test, lstm_predictions, "LSTM")


# In[ ]:




