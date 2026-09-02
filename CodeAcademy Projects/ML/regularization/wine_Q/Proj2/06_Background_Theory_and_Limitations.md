# Background Theory: Regularization in Machine Learning

*Companion document to the Wine Quality Regularization lab notebooks.*

## 1. Why regularize at all?

Every predictive model faces the same question: how well does it generalize from data it has seen to data it hasn't? A model can achieve very low error on its training data simply by growing complex enough to memorize noise rather than learn signal. This is **overfitting**, and it shows up as:

- Training performance much better than test performance.
- High model complexity (many parameters relative to data volume).
- Coefficients that respond to multicollinear features by splitting credit unstably between them.
- Coefficients that appear to be fitting noise rather than a genuine relationship.

Regularization addresses this by modifying the *objective* the model optimizes, rather than the data itself:

```
Total Loss(β) = Data Loss(β) + α · Penalty(β)
```

Because the optimizer must now minimize the sum, it can no longer chase a perfect fit to the training data for free — every unit of coefficient magnitude costs something. This deliberately raises training error in exchange for lower variance on unseen data. It is applied **during** model fitting, as part of the objective function — not as a preprocessing step and not as feature engineering.

## 2. The bias-variance tradeoff

- **Variance** is a model's sensitivity to the particular training sample it saw — a high-variance model changes a lot if you retrain it on a slightly different sample of the same population. Overfitting is a high-variance symptom.
- **Bias** is systematic error from a model being too constrained to capture the true relationship. Underfitting is a high-bias symptom.

Regularization trades one for the other: increasing `α` increases bias (the model is more constrained, so training error rises) and decreases variance (the model is less sensitive to the specific training sample, so it should generalize better) — up to a point. Push `α` too far and the model becomes so constrained it underfits everything, and both training and test performance degrade. The goal of hyperparameter tuning is to find the sweet spot, not to maximize regularization.

## 3. The two penalty families

For a two-feature linear model `y = b0 + b1x1 + b2x2`, the unregularized loss is minimized by Ordinary Least Squares (or gradient descent for larger problems). Regularization adds a penalty term:

| | Penalty | Constraint surface | Behavior |
|---|---|---|---|
| **L1 (Lasso)** | `α·(\|b1\| + \|b2\|)` | Diamond (sharp corners on the axes) | Coefficients can be driven to *exactly* zero |
| **L2 (Ridge)** | `α·(b1² + b2²)` | Circle (smooth boundary) | Coefficients shrink toward zero but (almost) never reach it |

Geometrically, minimizing the new loss means finding the point on (or inside) the constraint surface closest to the unconstrained optimum. Because L1's surface has corners exactly on the axes, the loss function's elliptical contours are likely to first touch the constraint at a corner — setting one coordinate to zero. L2's smooth circular boundary has no such corners, so the touching point almost never lands exactly on an axis; instead every coefficient shrinks a little.

This is why **L1 performs feature selection** (an *embedded* selection method, since it happens inside model fitting rather than as a separate pass over the data) and **L2 only performs shrinkage**. Elastic Net blends both penalties (`α₁·Σ|bⱼ| + α₂·Σbⱼ²`, controlled by a mixing parameter `l1_ratio`) to get some sparsity while being less sensitive than pure L1 to which of several correlated features gets kept.

**A structural detail worth remembering:** the intercept `b0` is never penalized by either method. Penalizing it would make the "best" model depend on an arbitrary shift in the target's scale (e.g., Celsius vs. Fahrenheit), which is not a property we want a *complexity* penalty to have. Regularization constrains the *slopes* — how much each feature matters — not the model's baseline output.

## 4. Hyperparameters vs. parameters

- **Parameters** (the `b`'s / coefficients) are learned from data during fitting.
- **Hyperparameters** (`alpha`, or `C` for `LogisticRegression`) are chosen *before* fitting and control how fitting behaves. They are not learned from a single fit; they are searched for via cross-validation.

`LogisticRegression` parameterizes regularization strength inversely, as `C = 1/alpha`: a *small* `C` means *strong* regularization (opposite direction from `alpha`), which is a common source of confusion and one of the "classic exam traps" in this material.

## 5. Hyperparameter tuning and cross-validation

Since the "right" amount of regularization depends on the data, it's found empirically: sweep a grid of candidate values, evaluate each with **k-fold cross-validation** (the data is split into k folds; each fold takes a turn as the held-out validation set while the rest train the model), and pick the value with the best average validation score. `GridSearchCV` automates this — a search over 100 candidate values with 5-fold CV performs 500 model fits.

Because the effect of `alpha` on the penalty differs by an order of magnitude between L1 (linear penalty growth) and L2 (quadratic penalty growth), search grids should typically be **log-spaced** (`np.logspace`) rather than linear, and Ridge's useful range of `alpha` values is usually several orders of magnitude wider than Lasso's.

## 6. Limitations of this style of lab and this class of method

**Limitations of linear regularization itself:**

- **Linearity assumption.** Ridge, Lasso, and regularized logistic regression only ever mix features linearly (or, for logistic regression, linearly inside a sigmoid). If the true relationship between features and target is strongly non-linear or involves feature interactions, no amount of regularization will fix that — you'd need polynomial/interaction features, kernel methods, trees, or neural networks, each of which has its own analogous-but-different regularization tools (max depth/min samples for trees, dropout/weight decay/early stopping/batch normalization for neural networks).
- **Regularization fixes variance, not bias from a wrong model family, and not data-quality problems.** It cannot compensate for label noise, sampling bias, leakage between train and test sets, or missing an important variable entirely.
- **L1's feature "selection" is not causal inference.** Among correlated features, which one L1 zeroes out can be somewhat arbitrary and sensitive to the specific sample or random seed — a zeroed coefficient means "redundant given what else is in the model," not "unrelated to the outcome."
- **Coefficient magnitude ≠ importance in absolute units unless features are scaled identically** — always scale before comparing or ranking coefficients.
- **Choice of hyperparameter range matters.** An `alpha`/`C` grid that doesn't bracket the true optimum will report a "best" value that's just the best of a bad set of options; always check whether the winning value sits at the edge of your search range (a sign you should widen it) or comfortably in the interior.
- **Cross-validated hyperparameters can still be optimistic** if the *same* held-out test set is reused for both hyperparameter selection and final reporting — this lab's L1 example (fit on `X, y` in full) illustrates exactly this trap. Nested cross-validation is the fuller fix.

**What this type of lab simulates successfully:**

- The mechanics of overfitting: comparing train vs. test performance on a real, tabular, moderately-sized dataset.
- The geometric intuition and practical difference between L1 and L2 penalties, made visible by comparing coefficient bar charts before and after regularization.
- The mechanics of hyperparameter tuning: coarse manual search → fine automated search via `GridSearchCV` or the `*CV` estimator family.
- The bias-variance tradeoff, observable directly as training error rises and test error initially falls as `alpha` increases (or `C` decreases).
- Transferable, dataset-agnostic scikit-learn patterns (`StandardScaler` → split → fit → `GridSearchCV` → evaluate) that generalize to most tabular regression/classification problems — this is exactly what the reusable template notebook packages up.

**What this type of lab cannot simulate or teach on its own:**

- Regularization behavior in non-linear models (tree-based min-samples/depth pruning, or neural-network dropout/weight decay/early stopping) — the intuition transfers loosely ("a penalty trades fit for simplicity") but the mechanisms and failure modes differ substantially.
- Behavior on very high-dimensional or sparse data (text, genomics, images) where L1's feature-selection properties are put to much heavier use and where computational considerations (solver choice, `saga` vs. `liblinear`) become dominant rather than incidental.
- Real deployment concerns: data drift after deployment, monitoring, retraining cadence, and fairness/bias auditing of a fielded model — regularization only ever addresses overfitting on data that resembles what the model was trained on.
- Causal questions ("does higher alcohol content *cause* higher perceived quality?") — regularized regression coefficients describe association, not causation, and a coefficient shrinking or vanishing under L1 says nothing about causal relevance.

## 7. Suggested extensions once the lab is comfortable

1. Repeat the classification workflow as a **regression** problem by predicting `alcohol` or another continuous column instead of the binarized `quality`, using `Ridge`/`Lasso`/`ElasticNet` directly — the reusable template notebook already supports this via its `TASK` flag.
2. Add **learning curves** (training/validation score vs. training-set size) to distinguish "needs more data" from "needs more/less regularization."
3. Implement **nested cross-validation** to get an unbiased estimate of generalization performance after hyperparameter tuning.
4. Compare regularized linear models against a **tree-based baseline** (e.g., `RandomForestClassifier`) to see how much of the wine-quality signal is genuinely linear versus not.
5. Explore **`SequentialFeatureSelector`** or **RFE** (wrapper/embedded feature selection alternatives mentioned in the lesson) and compare the feature subsets they choose to L1's.
