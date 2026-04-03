

This project calibrates drifting PM2.5 air quality sensors by training machine learning models against "truth" sensor data. It compares the performance of Linear Regression, Random Forests, and a Deep Neural Network (MLP).

## Features
- **Temporal Feature Engineering**: Extracts hour and month to account for environmental cycles.
- **Multivariate Scaling**: Uses Min-Max scaling for normalized model training.
- **Model Comparison**: Evaluates RMSE, and R² scores.
- **Automated Visualization**: Generates learning curves and scatter plots comparing model fit against ground truth.

## Installation & Usage

1. Clone the repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git](https://github.com/kaviudari99-jpg/Deep-Learning-Based-Sensor-Calibration---Drift-Compensation)
   cd https://github.com/kaviudari99-jpg/Deep-Learning-Based-Sensor-Calibration---Drift-Compensation
   
