
# AgriAI Decision Support Assistant
## Explainability Analysis Report

### 1. Introduction

This report presents an exploratory explainability analysis
of Random Forest crop-yield prediction models for beans,
maize and potatoes in Kenya.

The objective is to identify which input variables
the fitted models rely on when making predictions.

### 2. Methodology

Permutation feature importance was used.

Each feature was randomly shuffled 30 times, and the
resulting change in model performance was measured
using negative mean absolute error.

The models were evaluated using a chronological
80:20 training and testing split.

The test period covered 2011–2023, giving 13 observations.

### 3. Beans

The previous-year bean yield was the dominant predictor,
with permutation importance of 9.790.

Other influential features included:
- Potash use per area: 0.178
- Nitrogen use per area: 0.044
- Total agricultural potash use: 0.039
- Nitrogen manure applied: 0.015

The model relied heavily on historical bean yield.

### 4. Maize

Temperature change was the leading predictor,
with permutation importance of 1.249.

Other influential features included:
- Nitrogen manure applied: 0.367
- Phosphorus manure applied: 0.152
- Total agricultural phosphate use: 0.136
- Potassium manure applied: 0.113

Unlike beans and potatoes, temperature change
was the leading feature in the maize model.

### 5. Potatoes

Previous-year potato yield was the dominant predictor,
with permutation importance of 322.477.

Other influential features included:
- Phosphate use per area: 0.596
- Phosphorus crop residue removal: 0.477
- Nitrogen use per area: 0.301
- Total agricultural phosphate use: 0.212

The model relied predominantly on historical
potato yield.

### 6. Comparative observations

The leading predictors differed across crops:

- Beans: Previous-year bean yield.
- Maize: Temperature change.
- Potatoes: Previous-year potato yield.

These findings suggest that crop-specific models
may be useful when developing agricultural
decision-support systems.

### 7. Limitations

The dataset contains 62 annual observations,
with only 13 observations in the test period.

Permutation importance estimates may therefore
be unstable.

The results measure model dependence and do not
establish causal relationships between agricultural
inputs and crop yields.

The numerical importance scores should not be
compared directly across crops because their
yield scales differ.

Further validation using larger datasets,
additional geographical observations and
alternative validation strategies is recommended.

### 8. Conclusion

The explainability analysis demonstrates that
different variables influence the predictions
of the three crop-specific Random Forest models.

Historical yield dominates the bean and potato
models, while temperature change is the leading
feature for maize.

These preliminary findings provide a foundation
for developing a transparent agricultural
decision-support assistant.

---
AgriAI Decision Support Assistant
Research and development portfolio project.
