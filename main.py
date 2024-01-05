
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




df = pd.read_csv('fullData.csv', index_col=0)
df.index = pd.to_datetime(df.index)
print(df.head())

# df = df.resample('3H').agg({
#     'SnowGain': np.sum,
#     'AGDTemp': np.mean,
#     'AGDHumid': np.mean,
#     'AGDWindSpeed': np.mean,
#     'AGDWindDir': np.median,
#     'AGDWindGust': np.max,
#     'KHCRPress': np.mean,
#     'KHCRTemp': np.mean,
#     'KHCRHumid': np.mean,
#     'KHCRWind': np.mean,
#     'KHCRCloud3': np.mean,
#     'KHCRCloud1': np.mean,
#     'KHCRCloud2': np.mean,
#     'KHCRVis': np.mean,
#     'F8379Press': np.mean,
#     'F8379Temp': np.mean,
#     'F8379Humid': np.mean,
#     'F8379Wind': np.mean,
#     'KU42Press': np.mean,
#     'KU42Temp': np.mean,
#     'KU42Humidity': np.mean,
#     'KU42Wind': np.mean,
#     'KU42Cloud3': np.mean,
#     'KU42Cloud1': np.mean,
#     'KU42Cloud2': np.mean,
#     'KU42Vis': np.mean,
#     'KPVUPress': np.mean,
#     'KPVUTemp': np.mean,
#     'KPVUHumid': np.mean,
#     'KPVUWind': np.mean,
#     'KPVUCloud3': np.mean,
#     'KPVUCloud1': np.mean,
#     'KPVUCloud2': np.mean,
#     'KPVUVis': np.mean,
#     'KSLCPress': np.mean,
#     'KSLCTemp': np.mean,
#     'KSLCHumid': np.mean,
#     'KSLCWind': np.mean,
#     'KSLCCloud3': np.mean,
#     'KSLCCloud1': np.mean,
#     'KSLCCloud2': np.mean,
#     'KSLCVis': np.mean
# })

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
# scaled_data = df.values

# Define sequence length and features
sequence_length = 5  # Number of time steps in each sequence
num_features = len(df.columns)


# Create sequences and corresponding labels
sequences = []
labels = []
for i in range(len(scaled_data) - sequence_length):
    seq = scaled_data[i:i+sequence_length]
    label = scaled_data[i+sequence_length][0]  
    sequences.append(seq)
    labels.append(label)

# Convert to numpy arrays
sequences = np.array(sequences)
labels = np.array(labels)
print("Sequences shape:", sequences.shape)
print("Labels shape:", labels.shape)


train_x = sequences[int(0.2*len(sequences)):]
train_y = labels[int(0.2*len(labels)):]
test_x = sequences[:int(0.2*len(sequences))]
test_y = labels[:int(0.2*len(labels))]




print("Train X shape:", train_x.shape)
print("Train Y shape:", train_y.shape)
print("Test X shape:", test_x.shape)
print("Test Y shape:", test_y.shape)


# Create the LSTM model
model = Sequential()

# Add LSTM layers with dropout
model.add(LSTM(units=128, input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=64, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=32, return_sequences=False))
model.add(Dropout(0.2))

# Add a dense output layer
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# print(model.summary())


# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model_weightsV5.h5', monitor='val_loss', save_best_only=True)

# Train the model
history = model.fit(
    train_x, train_y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,  # Use part of the training data as validation
    callbacks=[early_stopping, model_checkpoint]
)

best_model = tf.keras.models.load_model('best_model_weights.h5')
test_loss = best_model.evaluate(test_x, test_y)
print("Test Loss:", test_loss)
print(best_model.summary())
print(len(best_model.layers[0].get_weights()))
    


# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.show()



from sklearn.metrics import mean_absolute_error, mean_squared_error

# Assuming you have trained the model and have the 'best_model' object
# Also, 'test_x' and 'test_y' should be available

# Predict temperatures using the trained model
predictions = best_model.predict(test_x)

# Calculate evaluation metrics
mae = mean_absolute_error(test_y, predictions)
mse = mean_squared_error(test_y, predictions)
rmse = np.sqrt(mse)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)



# y_true values
test_y_copies = np.repeat(test_y.reshape(-1, 1), test_x.shape[-1], axis=-1)
true_temp = scaler.inverse_transform(test_y_copies)[:,0]
# true_temp = test_y_copies[:,0]


# predicted values
prediction = best_model.predict(test_x)
prediction_copies = np.repeat(prediction, 42, axis=-1)
predicted_snow = scaler.inverse_transform(prediction_copies)[:,0]
# predicted_snow = prediction_copies[:,0]


# Plotting predicted and actual temperatures
plt.figure(figsize=(10, 6))
plt.plot(df.index[:600], true_temp[:600], label='Actual')
plt.plot(df.index[:600], predicted_snow[:600], label='Predicted')
plt.title('Snow Prediction vs Actual')
plt.xlabel('Time')
plt.ylabel('Snow')
plt.legend()
plt.show()
     