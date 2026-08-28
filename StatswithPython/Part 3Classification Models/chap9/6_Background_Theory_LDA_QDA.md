# Background & Theory: Bayes' Theorem, LDA, and QDA

*Companion reading for the Discriminant Analysis lab series (based on Chapter 9 of Building
Statistical Models in Python).*

## 1. Why this chapter starts with probability

Classification, at its core, is a question about probability: given what we observe about a case
(its features, `X`), how likely is it to belong to each possible class, `Y = k`? Logistic
regression answers this by modeling `P(Y=k | X=x)` directly as a function of `X`. LDA and QDA take
a different route: they model how `X` is *distributed within each class*, then use Bayes' Theorem
to flip that around into `P(Y=k | X=x)`. This is why they're called **generative** classifiers —
they model how the data could have been generated, class by class — as opposed to **discriminative**
classifiers like logistic regression, which model the boundary directly without describing each
class's distribution.

## 2. Bayes' Theorem, in plain terms

Bayes' Theorem answers: given some new evidence, how should I update a belief I already had?

```
P(B | A) = P(A | B) * P(B) / P(A)
```

- `P(B)` is the **prior** — what you believed before seeing the evidence.
- `P(B | A)` is the **posterior** — what you believe after seeing evidence `A`.
- `P(A | B)` is the **likelihood** — how probable the evidence is, assuming `B` is true.
- `P(A)` is a normalizing constant so the posterior is a valid probability.

The medical-test example in the source chapter is the clearest illustration: even a 95%-accurate
test with a 30% false-positive rate only moves a 20% prior up to about 44% after one positive
result, because false positives are common enough to muddy a single test's signal. Only repeated,
independent positive results converge you toward certainty. This "slow convergence under noisy
evidence" behavior is worth internalizing — it's the same mechanism at work, in miniature, every
time LDA or QDA computes a posterior probability for a data point sitting near a decision
boundary: points near the boundary get soft, unconfident posteriors (close to 0.5 in the binary
case) for exactly the same reason a single noisy medical test doesn't fully convince you.

## 3. From Bayes' Theorem to a classifier

For a classification problem, Bayes' Theorem takes the shape:

```
P(Y=k | X=x)  =  P(X=x | Y=k) * P(Y=k)  /  P(X=x)
```

Here:
- `P(Y=k)` is the **prior** probability of class `k` — usually just the observed class frequency.
- `P(X=x | Y=k)` is the **class-conditional density** — how the features are distributed *within*
  class `k`. This is the part LDA and QDA model explicitly, and where they differ from each other.
- `P(X=x)` is a normalizing constant, the same for every class, so in practice you only need to
  compare the numerators across classes to pick the most likely one.

Both LDA and QDA assume `P(X=x | Y=k)` is a **multivariate Gaussian** distribution with a
class-specific mean vector `μ_k`. They differ in what they assume about the covariance:

- **LDA** assumes every class shares one covariance matrix, `Σ`. Because the quadratic terms in
  the Gaussian exponent then cancel out when you compare classes, the resulting decision boundary
  is **linear** in `x` — hence "Linear" Discriminant Analysis.
- **QDA** allows each class its own covariance matrix, `Σ_k`. The quadratic terms don't cancel,
  so the boundary is a genuine **quadratic** curve (an ellipse, parabola, or hyperbola, depending
  on the classes' geometry) — hence "Quadratic" Discriminant Analysis.

This single difference explains almost everything about how the two models behave differently in
practice.

## 4. The assumptions, and what they actually require of your data

1. **Gaussian class-conditional distributions.** Every input feature, within each class, should
   look roughly bell-shaped. This is the assumption most often stretched in practice — count data,
   ordinal survey scales, and one-hot-encoded categoricals are not truly Gaussian, but the models
   often still work reasonably well if the departure is mild and there's enough data to average
   out the noise. It's worth explicitly checking (visually, and with a normality test used as a
   signal rather than a strict gate) rather than assuming it.

2. **Equal covariance across classes (LDA only).** This is the assumption that separates LDA from
   QDA. It means the *shape and orientation* of the feature-space "cloud" for each class should be
   similar — not necessarily the same size in absolute terms after scaling, but the pattern of
   variances and pairwise correlations between features should look similar class to class. When
   this assumption clearly fails (visibly different spreads/orientations between classes), QDA is
   the natural next step, at the cost of needing more data to estimate the extra covariance
   matrices reliably.

3. **Independent observations.** Both models assume rows are independent draws. Repeated
   measurements on the same subject, time-series structure, or clustered sampling (e.g. multiple
   students from the same classroom) all violate this and can make the estimated covariance
   structure — and therefore the decision boundary — misleadingly confident. This assumption is
   about your *data collection design*, not something you can check by looking at a single
   dataframe; it needs to be reasoned about from how the data was gathered.

## 5. The bias-variance tradeoff, made concrete

LDA and QDA are a rare, clean illustration of the bias-variance tradeoff because they are
literally the *same family* of model with one assumption relaxed:

- LDA has **higher bias** (it can only ever draw a straight decision boundary, which is wrong if
  the true class geometries differ) but **lower variance** (it estimates far fewer parameters — one
  shared covariance matrix instead of one per class — so its estimate is more stable with limited
  data).
- QDA has **lower bias** (it can curve to match each class's true shape) but **higher variance**
  (each class's covariance matrix has `O(p²)` free parameters, where `p` is the number of
  features; with `K` classes that's `K` times as many parameters to estimate as LDA, and each is
  estimated from only that class's subset of the data).

The practical consequence: QDA needs meaningfully more data *per class* than LDA to reach its
theoretical advantage. With few observations in a class, QDA's covariance estimate becomes noisy
or even singular (non-invertible), which shows up as suspiciously perfect training accuracy and
poor test accuracy — classic overfitting.

## 6. Supervised dimensionality reduction with LDA

Beyond classification, LDA can be used to project data onto a small number of axes that maximize
*class separation* rather than *overall variance* (which is what PCA maximizes instead). With `K`
classes, LDA yields at most `K − 1` useful discriminant axes. This is a genuinely different
technique from PCA and can be a strong preprocessing step or a way to visualize/summarize
class separation in high dimensions — but note it requires the same Gaussian and equal-covariance
assumptions as LDA classification, and it "bakes in" the target label, so it's a supervised
technique that shouldn't be used if you plan to later evaluate against the same labels without a
held-out split.

## 7. Limitations — what these models cannot do

- **Anything beyond an ellipse or a straight line.** If the true boundary between classes is a
  wiggly, non-convex shape, neither LDA nor QDA can represent it, no matter how much data you add.
  That calls for kernel methods, tree ensembles, or neural networks.
- **Robustness to outliers.** Means and covariances are not robust statistics — a handful of
  extreme points can noticeably shift a fitted boundary. Consider outlier screening or robust
  covariance estimators if this is a concern.
- **High-dimensional, low-sample-size settings.** When the number of features approaches or
  exceeds the number of observations per class, covariance matrices become singular or unstable.
  `LinearDiscriminantAnalysis` offers `shrinkage` for exactly this scenario; QDA has no equally
  simple fix and generally needs more data or feature reduction first.
- **Non-Gaussian feature types.** Sparse, highly skewed, purely categorical, or text/image data
  don't fit the Gaussian assumption well. Feeding one-hot-encoded categorical columns into
  LDA/QDA (as the chapter's `affairs` example does) is a common, generally-tolerated
  approximation — not a textbook-clean use of the method, and worth being explicit about when you
  do it.
- **Calibration under class imbalance.** The prior `π_k` is estimated from observed class
  frequency by default. If your training data's class balance doesn't reflect the real-world
  balance you'll deploy against, override `priors` explicitly, or your posteriors will be
  systematically biased toward whichever class was more common in training.

## 8. What this class of lab can simulate successfully

- **Tabular problems with a modest number of continuous, roughly Gaussian, per-class features** —
  biometric measurements, sensor readings, standardized survey scales, chemical assay results
  (the classic UCI Wine dataset is a textbook example where LDA does very well).
- **Situations where you want class-conditional density estimates**, not just a hard decision —
  e.g. generating synthetic samples per class, or computing calibrated class probabilities for
  downstream decision-making.
- **Teaching the bias-variance tradeoff concretely**, since LDA and QDA differ by exactly one
  assumption, letting you isolate its effect (as in the synthetic unequal-covariance example in
  the extended lab) in a way most model comparisons can't offer this cleanly.
- **Fast, interpretable baselines** for any classification task, before reaching for
  more complex models — both fit almost instantly and their coefficients/boundaries are directly
  interpretable.

## 9. What it cannot simulate successfully

- High-dimensional, sparse, or highly non-Gaussian data (raw text, images, complex categorical
  interactions) — technically computable, but the Gaussian assumption is stretched far enough that
  results are unreliable; other model families are more appropriate.
- Problems with a genuinely non-elliptical decision boundary — no amount of correct
  preprocessing will let LDA/QDA represent, say, a checkerboard pattern of classes in feature
  space.
- Very rare classes with too little data to estimate even a single covariance matrix reliably
  (QDA especially) — expect instability and misleadingly optimistic training performance.

## 10. Where the lab series takes this further than the book

The two source notebooks (`2_Linear_Discriminant_Analysis.ipynb`,
`3_Quadratic_discriminant_analysis.ipynb`) demonstrate the mechanics but stop short of validating
the assumptions numerically, comparing against a baseline, or checking whether results generalize
beyond a single train/test split. The extended lab in this series adds:

- Numerical + visual checks of the normality and equal-covariance assumptions (Shapiro-Wilk,
  covariance-matrix comparison) instead of asserting them.
- A fix for a data-leakage bug in the original code (fitting a second `StandardScaler` on the test
  set instead of reusing the one fit on training data).
- ROC/AUC evaluation alongside precision/recall, and 5-fold cross-validation to sanity-check the
  single-split results.
- A synthetic dataset built specifically to isolate *why* QDA can outperform LDA — something
  neither source notebook demonstrates directly, since both real datasets used only happen to
  favor one model without a clean counter-example to compare against.
- A side-by-side comparison against Logistic Regression as an external baseline, since discriminant
  analysis should always be judged against at least one model outside its own family.

## Further reading
- *An Introduction to Statistical Learning* (James, Witten, Hastie, Tibshirani) — Chapter 4 covers
  LDA/QDA derivations at a similar level with additional worked examples.
- scikit-learn's [Linear and Quadratic Discriminant Analysis user guide](https://scikit-learn.org/stable/modules/lda_qda.html)
  for solver options (`svd`, `lsqr`, `eigen`) and the `shrinkage` parameter.
