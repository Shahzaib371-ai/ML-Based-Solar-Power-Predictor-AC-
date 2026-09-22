from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

theta = np.load(os.path.join(RESULTS_DIR, "theta_B.npy"))
mean = np.load(os.path.join(RESULTS_DIR, "scaler_B_mean.npy"))
std = np.load(os.path.join(RESULTS_DIR, "scaler_B_std.npy"))


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        # ---- User inputs from the form ----
        sw_radiation = float(request.form["sw_radiation"])
        temp_2m = float(request.form["temp_2m"])
        cloud_cover = float(request.form["cloud_cover"])
        hour = float(request.form["hour"])

        # ---- Cyclical hour encoding (must match training) ----
        sin_hour = np.sin(2 * np.pi * hour / 24.0)
        cos_hour = np.cos(2 * np.pi * hour / 24.0)

        # ---- EXACT same order as features_B in the notebook ----
        # features_B = ['sw_radiation', 'temp_2m', 'cloud_cover', 'sin_hour', 'cos_hour']
        X = np.array([
            sw_radiation,
            temp_2m,
            cloud_cover,
            sin_hour,
            cos_hour
        ], dtype=float)

        if len(mean) != len(X) or len(std) != len(X):
            return (
                f"Scaler mismatch! "
                f"X={len(X)}, mean={len(mean)}, std={len(std)}"
            )

        # ---- Standardize + add intercept ----
        X_scaled = (X - mean) / std
        X_final = np.insert(X_scaled, 0, 1.0)

        # ---- Predict ----
        prediction = np.dot(X_final, theta)
        prediction = f"{prediction:.2f}"

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)