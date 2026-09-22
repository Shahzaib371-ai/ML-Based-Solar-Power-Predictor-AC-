# Solar Plant AC Power Predictor | Anti-Gravity ML Engine

An advanced machine learning web application built with **Flask**, **NumPy**, and custom **Glassmorphism CSS**, designed to predict hourly AC power output from solar power plants using weather telemetry and temporal mapping.

## Project Overview

Accurately predicting solar power generation is important for efficient grid management and renewable energy integration.

This project implements a trained machine learning regression model using **Set B parameters** to estimate AC power output based on atmospheric conditions and time-based features.

The trained model is integrated into a lightweight Flask web application that provides an interactive dashboard for solar power prediction.

## Key Features

* **Anti-Gravity UI/UX**: High-tech dark mode interface with deep-space gradients, frosted glassmorphism cards, and neon blue accents.
* **Interactive Telemetry Inputs**: Sliders and number inputs for adjusting environmental parameters.
* **Dynamic 24-Hour Power Profile**: Generates an estimated power profile across a complete 24-hour cycle.
* **Machine Learning Inference**: Uses trained model weights and feature scaling parameters stored in `.npy` files.
* **Fast Prediction**: Uses NumPy-based numerical computation without requiring a heavyweight ML framework during inference.
* **Power Visualization**: Displays predicted AC power and capacity utilization through an interactive dashboard.

## Project Structure

```text
ML-Based-Solar-Power-Predictor-AC-/
├── app.py
├── templates/index.html
├── static/style.css
├── results/theta_B.npy
├── results/scaler_B_mean.npy
├── results/scaler_B_std.npy
├── ML_Project.ipynb
├── view_npy.py
├── solar_power_prediction_blog (1).md
├── assign task.pdf
└── README.md
```

## File Description

### app.py

Main Flask application containing the web routes and machine learning inference logic.

### index.html

Jinja2 HTML template containing the main solar power prediction dashboard.

### style.css

Custom CSS file containing the Glassmorphism interface and dashboard styling.

### theta_B.npy

Contains the trained regression model weights.

### scaler_B_mean.npy

Contains the feature mean values used for Z-score normalization.

### scaler_B_std.npy

Contains the feature standard deviation values used for Z-score normalization.

### ML_Project.ipynb

Jupyter Notebook containing data analysis, preprocessing, model training, evaluation, and generation of the trained model parameters.

### view_npy.py

Helper script to inspect the contents of the `.npy` model files.

### solar_power_prediction_blog (1).md

Companion blog write-up describing the project.

### assign task.pdf

Project assignment/task document.

## Tech Stack

| Layer                 | Technology            |
| --------------------- | --------------------- |
| Web Application       | Flask                 |
| Template Engine       | Jinja2                |
| Frontend              | HTML, CSS, JavaScript |
| Numerical Computation | NumPy                 |
| Model Training        | Jupyter Notebook      |
| Model                 | Linear Regression     |
| Model Artifacts       | NumPy `.npy` files    |
| Programming Language  | Python 3.x            |

## Installation

### 1. Navigate to the Project Directory

```bash
cd ML-Based-Solar-Power-Predictor-AC-
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install flask numpy joblib
```

### 5. Verify Model Files

Make sure the following files exist in the `results/` directory:

```text
theta_B.npy
scaler_B_mean.npy
scaler_B_std.npy
```

If these files are missing, open `ML_Project.ipynb` and run the training and model-export cells.

## Running the Application

From the project root directory, run:

```bash
python app.py
```

After starting Flask, open the following address in your web browser:

```text
http://127.0.0.1:5000
```

On Replit, run the **Start application** workflow and use the webview instead. Replit dependencies are recorded in `pyproject.toml`.

## Using the Dashboard

The dashboard allows the user to:

1. Adjust the **Hour of Day**.
2. Adjust **Shortwave Radiation**.
3. Adjust **2m Air Temperature**.
4. Adjust **Cloud Cover**.
5. Click **Predict AC Power**.
6. View the predicted AC power in **kW**.
7. View the predicted AC power in **MW**.
8. View the **Capacity Utilization** percentage.
9. View the predicted **24-Hour Power Profile**.

## Machine Learning Model

The model uses **Z-score standardization** followed by a linear regression prediction.

The standardization process is:

```text
Z_i = (X_i - μ_i) / σ_i
```

The final prediction is calculated using:

```text
ŷ = θ_0 + Σ(θ_i × Z_i)
```

Where:

* `X_i` = Raw input feature.
* `μ_i` = Mean of the feature calculated from the training dataset.
* `σ_i` = Standard deviation of the feature calculated from the training dataset.
* `θ_i` = Learned model weight.
* `ŷ` = Predicted AC power output in kW.

The input features are first standardized using the stored mean and standard deviation values.

The normalized features are then multiplied by the trained model weights to calculate the predicted AC power output.

## 24-Hour Power Profile

The application generates a predicted power profile for a complete 24-hour period.

For each hour, the model estimates the expected AC power output using the selected environmental conditions and time-based features.

The predictions are then displayed through an interactive line chart, allowing the user to observe the expected variation in solar power generation throughout the day.

## Project Purpose

This project is developed as a **Final Year Project (FYP)** at the intersection of **Artificial Intelligence** and **Electrical Engineering**.

The main objective is to apply machine learning regression techniques to a real-world power systems problem: predicting solar AC power generation using weather telemetry and temporal information.

The project demonstrates how machine learning can be integrated with electrical engineering applications for renewable energy forecasting and power system analysis.

## Future Improvements

Possible future improvements include:

* Integration with real-time weather APIs.
* Integration with live solar plant telemetry.
* Deployment on cloud infrastructure.
* Comparison with advanced machine learning models.
* Improved prediction accuracy using additional weather features.
* Historical prediction analysis.
* Database integration for storing predictions.
* Real-time monitoring and visualization.
* Deployment on edge computing hardware.

## Author

**Shahzaib Hasnain**

FAST NUCES, CFD Campus

## License

This project is developed for academic purposes as part of a Final Year Project.

It may be used and adapted for educational and learning purposes.
