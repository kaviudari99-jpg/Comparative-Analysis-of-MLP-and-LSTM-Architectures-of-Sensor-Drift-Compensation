import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # --- 1. DATA LOADING & LOG-TRANSFORM ---
    path = os.path.join('data', 'Measurement_info.csv')
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: Data file not found.")
        return

    pm25 = df[df['Item code'] == 9].copy()
    pm25['Measurement date'] = pd.to_datetime(pm25['Measurement date'])
    
    # Cyclical Encoding
    pm25['H_sin'] = np.sin(2 * np.pi * pm25['Measurement date'].dt.hour / 24)
    pm25['H_cos'] = np.cos(2 * np.pi * pm25['Measurement date'].dt.hour / 24)

    truth = pm25[pm25['Instrument status'] == 0].groupby('Measurement date')['Average value'].mean().reset_index()
    drift = pm25[pm25['Instrument status'] == 1][['Measurement date', 'Average value', 'H_sin', 'H_cos']]
    pairs = pd.merge(truth, drift, on='Measurement date').dropna()

    # --- CRITICAL STEP: Log Transform the Target to catch peaks ---
    y_raw = pairs[['Average value_x']].values
    y_log = np.log1p(y_raw) # log(1 + x)
    
    scaler_x, scaler_y = MinMaxScaler(), MinMaxScaler()
    X_scaled = scaler_x.fit_transform(pairs[['Average value_y', 'H_sin', 'H_cos']].values)
    y_scaled = scaler_y.fit_transform(y_log)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    X_train_lstm = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_test_lstm = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    # --- 2. MODEL DEFINITIONS ---
    
    # Early Stopping Callback: Prevents the "Orange Line Drift"
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Basic MLP
    mlp_basic = tf.keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # Deep MLP: Added L2 Regularization to stop overfitting
    mlp_deep = tf.keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(128, activation='relu', kernel_regularizer='l2'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    # LSTM: Increased capacity
    lstm_model = tf.keras.Sequential([
        layers.Input(shape=(1, 3)),
        layers.LSTM(100, return_sequences=True),
        layers.LSTM(50),
        layers.Dense(1)
    ])

    models = {"Basic MLP": (mlp_basic, X_train, X_test), 
            "Deep MLP": (mlp_deep, X_train, X_test), 
            "LSTM": (lstm_model, X_train_lstm, X_test_lstm)}
    
    plt.figure(figsize=(18, 10))

    for i, (name, (model, xt, xv)) in enumerate(models.items(), 1):
        print(f"Training {name}...")
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(xt, y_train, epochs=300, batch_size=32, 
                            validation_split=0.2, callbacks=[early_stop], verbose=0)
        
        # INVERSE LOG TRANSFORMATION
        preds_log = scaler_y.inverse_transform(model.predict(xv))
        preds_final = np.expm1(preds_log) # Reverse of log1p
        
        y_real = np.expm1(scaler_y.inverse_transform(y_test))
        r2 = r2_score(y_real, preds_final)

        # Plotting
        plt.subplot(2, 3, i)
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title(f'{name} (Best Epoch: {len(history.history["loss"])})')
        
        plt.subplot(2, 3, i+3)
        plt.scatter(y_real, preds_final, alpha=0.4, edgecolors='w')
        plt.plot([0, 110], [0, 110], 'r--') # Perfect fit line
        plt.title(f'{name} Fit - R2: {r2:.2f}')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()