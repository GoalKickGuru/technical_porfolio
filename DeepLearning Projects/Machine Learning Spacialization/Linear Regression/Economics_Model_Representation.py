"""
Economics Model Representation
Linear regression model f_w,b(x) = w*x + b for advertising → sales forecasting.
"""
import numpy as np
import matplotlib.pyplot as plt

def compute_model_output(x, w, b):
    """Loop-based model evaluation."""
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

def compute_model_output_vectorized(x, w, b):
    """Vectorized model evaluation."""
    return w * x + b

if __name__ == "__main__":
    x_train = np.array([1.0, 2.0])
    y_train = np.array([300.0, 500.0])
    w, b = 200.0, 100.0

    print("Training data:", x_train, y_train)
    print("Fitted model: sales = {:.0f} * ad_spend + {:.0f}".format(w, b))
    print("Predictions on training set:", compute_model_output_vectorized(x_train, w, b))

    x_new = 1.2
    forecast = w * x_new + b
    print(f"Forecast for ad spend {x_new}: {forecast:.1f} sales units "
          f"(≈ ${forecast * 10_000:,.0f})")
