"""
Credit Scoring with Logistic Regression – Model Representation
f_w,b(x) = sigmoid(w*x + b) for Probability of Default.
"""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_model_output(x, w, b):
    return sigmoid(w * x + b)

def predict(x, w, b, threshold=0.5):
    probs = compute_model_output(x, w, b)
    decisions = (probs >= threshold).astype(int)
    return probs, decisions

if __name__ == "__main__":
    x_train = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    y_train = np.array([0, 0, 0, 1, 1, 1])
    w, b = 2.0, -4.0

    print("DTI values     :", x_train)
    print("Actual default :", y_train)
    probs, decisions = predict(x_train, w, b, threshold=0.5)
    print("P(default)     :", np.round(probs, 3))
    print("Decisions      :", decisions)
    print(f"Decision boundary at DTI = {-b/w:.2f}")
    print(f"Training accuracy: {np.mean(decisions == y_train):.0%}")

    new_dti = np.array([0.8, 1.8, 2.7])
    new_p, new_d = predict(new_dti, w, b)
    print("\nNew applicants:")
    for dti, p, d in zip(new_dti, new_p, new_d):
        print(f"  DTI={dti:.1f} → P={p:.3f} → {'HIGH RISK' if d else 'LOW RISK'}")
