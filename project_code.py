"""
Sensor Calibration using Machine Learning
Compares Linear Regression, Random Forest, and a Deep Neural Network (MLP)
to calibrate drifting PM2.5 air quality sensor readings against truth data.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def main():
    # Initializing the Dataset and the Preprocessing
    path = os.path.join('data', 'Measurement_info.csv')
    
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"Error: Could not find data file at {path}.")
        return

    print("Data loaded successfully.")

    # Filtering PM2.5 from the dataset (Item code for PM2.5 is 9)
    pm25 = df[df['Item code'] == 9].copy()

    # Feature Engineering: Extract Temporal Proxies for Weather
    pm25['Measurement date'] = pd.to_datetime(pm25['Measurement date'])
    pm25['Hour'] = pm25['Measurement date'].dt.hour
    pm25['Month'] = pm25['Measurement date'].dt.month

    # Establish Truth (Status 0) and Drift (Status 1)
    truth = pm25[pm25['Instrument status'] == 0].groupby('Measurement date')['Average value'].mean().reset_index()
    truth.columns = ['Measurement date', 'Average value_Truth']
    
    drift = pm25[pm25['Instrument status'] == 1][['Measurement date', 'Average value', 'Hour', 'Month']]
    drift.columns = ['Measurement date', 'Average value_Drift', 'Hour', 'Month']

    # Merge into synchronous pairs
    pairs = pd.merge(truth, drift, on='Measurement date').dropna()

    # --- 2. MULTI-VARIABLE SCALING ---
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    # Input features: Drift reading, Hour, Month
    X_raw = pairs[['Average value_Drift', 'Hour', 'Month']].values
    # Target feature: Truth reading
    y_raw = pairs[['Average value_Truth']].values.reshape(-1, 1)

    X_scaled = scaler_x.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)

    # 80/20 Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    # --- 3. ALGORITHM EXECUTION ---
    
    # Alg 1: Linear Regression
    print("\nTraining Linear Regression...")
    lr = LinearRegression().fit(X_train, y_train)
    lr_preds_scaled = lr.predict(X_test)

    # Alg 2: Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train.flatten())
    rf_preds_scaled = rf.predict(X_test).reshape(-1, 1)

    # Alg 3: Deep Neural Network (MLP)
    print("Training Deep MLP (500 Epochs)...")
    mlp = tf.keras.Sequential([
        layers.Input(shape=(3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    mlp.compile(optimizer='adam', loss='mse')
    history = mlp.fit(X_train, y_train, validation_split=0.2, epochs=500, batch_size=64, verbose=0)
    mlp_preds_scaled = mlp.predict(X_test)

    # --- 4. INVERSE SCALING & METRIC CALCULATION ---
    # Bring predictions back to original ug/m3 scale for accurate metric calculation
    y_test_real = scaler_y.inverse_transform(y_test)
    lr_final = scaler_y.inverse_transform(lr_preds_scaled)
    rf_final = scaler_y.inverse_transform(rf_preds_scaled)
    mlp_final = scaler_y.inverse_transform(mlp_preds_scaled)

    def calculate_metrics(true, pred):
        mse = mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        return mse, mae, r2

    lr_metrics = calculate_metrics(y_test_real, lr_final)
    rf_metrics = calculate_metrics(y_test_real, rf_final)
    mlp_metrics = calculate_metrics(y_test_real, mlp_final)

    # --- 5. PERFORMANCE TABLE GENERATION ---
    data = {
        'Algorithm': ['Linear Regression', 'Random Forest', 'Deep Neural Network (MLP)'],
        'MSE': [lr_metrics[0], rf_metrics[0], mlp_metrics[0]],
        'MAE': [lr_metrics[1], rf_metrics[1], mlp_metrics[1]],
        'R2 Score': [lr_metrics[2], rf_metrics[2], mlp_metrics[2]]
    }

    comparison_table = pd.DataFrame(data)
    print("\n" + "="*75)
    print("             SENSORS CALIBRATION: MODEL PERFORMANCE SUMMARY")
    print("="*75)
    print(comparison_table.to_string(index=False))
    print("="*75)

    # Save the table to CSV for reports
    comparison_table.to_csv('sensor_calibration_results.csv', index=False)

    # --- 6. VISUALIZATION ENGINE ---
    plt.figure(figsize=(18, 10))

    # Plot 1: Neural Network Learning Curve
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Deep MLP Optimization Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()

    # Plots 2, 3, 4: Scatter Plot Fit Comparison
    models = [
        (lr_final, 'Linear Regression', 'gray'),
        (rf_final, 'Random Forest', 'green'),
        (mlp_final, 'Neural Network', 'blue')
    ]
    
    for i, (preds, title, color) in enumerate(models, 2):
        plt.subplot(2, 2, i)
        plt.scatter(y_test_real, preds, alpha=0.3, color=color)
        plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=2)
        plt.title(f'{title} Accuracy Fit')
        plt.xlabel('Ground Truth (ug/m³)')
        plt.ylabel('Calibrated Prediction')

    plt.tight_layout()
    plt.savefig('calibration_performance_plots.png', dpi=300)
    print("\nTask Complete: CSV saved and High-Res Plots generated.")
    plt.show()

if __name__ == '__main__':
    main()