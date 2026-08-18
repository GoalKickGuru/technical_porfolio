import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
cols = [
    "name",
    "landmass",
    "zone",
    "area",
    "population",
    "language",
    "religion",
    "bars",
    "stripes",
    "colours",
    "red",
    "green",
    "blue",
    "gold",
    "white",
    "black",
    "orange",
    "mainhue",
    "circles",
    "crosses",
    "saltires",
    "quarters",
    "sunstars",
    "crescent",
    "triangle",
    "icon",
    "animate",
    "text",
    "topleft",
    "botright",
]

df = pd.read_csv(
    "https://archive.ics.uci.edu/ml/machine-learning-databases/flags/flag.data",
    names=cols,
)

var = [
    "red",
    "green",
    "blue",
    "gold",
    "white",
    "black",
    "orange",
    "mainhue",
    "bars",
    "stripes",
    "circles",
    "crosses",
    "saltires",
    "quarters",
    "sunstars",
    "triangle",
    "animate",
]

# Task 1: Print country counts by landmass
print("--- Task 1: Count of Flags by Landmass ---")
print(df["landmass"].value_counts().sort_index())
print("\n" + "=" * 50 + "\n")

# ---------------------------------------------------------
# 2. Filter Dataset & Task Analysis
# ---------------------------------------------------------
# Task 2: Filter for Europe (3) and Oceania (6)
df_36 = df[df["landmass"].isin([3, 6])].copy()

# Task 3: Average predictor values by continent
print("--- Task 3: Average Predictor Values (Europe vs Oceania) ---")
print(df_36.groupby("landmass")[var].mean(numeric_only=True).T)
print("\n" + "=" * 50 + "\n")

# Task 4: Print predictor data types
print("--- Task 4: Predictor Variable Types ---")
print(df_36[var].dtypes)
print("\n" + "=" * 50 + "\n")

# Task 5: One-Hot Encode categorical predictors
labels = df_36["landmass"]
data = pd.get_dummies(df_36[var], columns=["mainhue"])

# Task 6: Train / Test Split
train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.4, random_state=1
)

# ---------------------------------------------------------
# 3. Hyperparameter Tuning by Depth
# ---------------------------------------------------------
# Task 7 & 8: Evaluate max_depth from 1 to 20
depths = range(1, 21)
acc_depth = []

for depth in depths:
  dt = DecisionTreeClassifier(random_state=10, max_depth=depth)
  dt.fit(train_data, train_labels)
  acc_depth.append(dt.score(test_data, test_labels))

# Task 9: Best max_depth
max_acc = np.max(acc_depth)
best_depth = depths[np.argmax(acc_depth)]
print(f"Task 9: Highest Accuracy by Depth = {max_acc * 100:.1f}%")
print(f"Optimal max_depth = {best_depth}\n")

# Task 10: Plot Decision Tree with Best Depth
plt.figure(figsize=(12, 6))
dt_depth = DecisionTreeClassifier(random_state=1, max_depth=best_depth)
dt_depth.fit(train_data, train_labels)
tree.plot_tree(
    dt_depth,
    feature_names=data.columns,
    class_names=["Europe", "Oceania"],
    filled=True,
)
plt.title(f"Decision Tree (max_depth={best_depth})")
plt.show()

# ---------------------------------------------------------
# 4. Hyperparameter Tuning by Pruning (ccp_alpha)
# ---------------------------------------------------------
# Task 11 & 12: Evaluate Cost Complexity Pruning parameters
acc_pruned = []
ccp_alphas = np.logspace(-3, 0, num=20)

for alpha in ccp_alphas:
  dt_prune = DecisionTreeClassifier(
      random_state=1, max_depth=best_depth, ccp_alpha=alpha
  )
  dt_prune.fit(train_data, train_labels)
  acc_pruned.append(dt_prune.score(test_data, test_labels))

# Task 13: Best ccp_alpha
max_acc_pruned = np.max(acc_pruned)
best_ccp = ccp_alphas[np.argmax(acc_pruned)]
print(f"Task 13: Highest Accuracy after Pruning = {max_acc_pruned * 100:.1f}%")
print(f"Optimal ccp_alpha = {best_ccp:.4f}\n")

# Task 14: Fit Final Pruned Tree
dt_final = DecisionTreeClassifier(
    random_state=1, max_depth=best_depth, ccp_alpha=best_ccp
)
dt_final.fit(train_data, train_labels)

plt.figure(figsize=(10, 5))
tree.plot_tree(
    dt_final,
    feature_names=data.columns,
    class_names=["Europe", "Oceania"],
    filled=True,
)
plt.title(f"Final Pruned Decision Tree (ccp_alpha={best_ccp:.4f})")
plt.show()