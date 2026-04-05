# Modeling Strategy: Higgsfield Churn Prediction (HackNU 2026)

## Table of Contents

1. [Problem Formulation & Framing Options](#1-problem-formulation--framing-options)
2. [Evaluation Strategy](#2-evaluation-strategy)
3. [Class Imbalance Handling](#3-class-imbalance-handling)
4. [Feature Selection & Preprocessing](#4-feature-selection--preprocessing)
5. [Individual Models & Hyperparameter Spaces](#5-individual-models--hyperparameter-spaces)
6. [Ensemble & Stacking Architectures](#6-ensemble--stacking-architectures)
7. [Hierarchical / Two-Stage Decomposition](#7-hierarchical--two-stage-decomposition)
8. [Explainability Layer](#8-explainability-layer)
9. [Pipelines to Test (Priority-Ordered)](#9-pipelines-to-test-priority-ordered)
10. [Experimental Protocol](#10-experimental-protocol)

---

## 1. Problem Formulation & Framing Options

We have a **3-class classification** task with classes `not_churned`, `vol_churn`, `invol_churn`. The label distribution is 50/25/25 — moderately imbalanced, with the minority classes (both churn types) equally represented but together matching the majority class.

### 1A. Direct Multiclass (Flat)

Train a single model with `objective = multi:softmax` or `multi:softprob`. All three classes compete simultaneously. This is the simplest approach and should serve as the **first baseline**.

**Advantages:** Simple, single model to tune, the model can learn shared features across churn types.  
**Disadvantages:** May underperform if vol_churn and invol_churn have very different feature importance profiles (which they almost certainly do — billing friction drives invol, engagement decay drives vol).

### 1B. Hierarchical Two-Stage Decomposition

Stage 1: Binary classifier — **churn vs not_churn**. This collapses vol + invol into one positive class.  
Stage 2: Binary classifier — **vol_churn vs invol_churn** (only applied to users predicted as churn by Stage 1).

**Advantages:** Each stage focuses on a cleaner separation. Stage 1 can leverage all churn signals broadly; Stage 2 specializes on the vol/invol distinction where feature profiles diverge sharply (billing features dominate invol; engagement features dominate vol). Easier to tune each stage. Error analysis is more interpretable.  
**Disadvantages:** Stage 2 errors compound on Stage 1 errors. Requires careful probability calibration between stages. Total pipeline is more complex to validate.

**Implementation detail:** For Stage 2, use *soft handoff* — don't threshold Stage 1 predictions. Instead, multiply Stage 1's churn probability by Stage 2's conditional vol/invol probabilities to get final 3-class probabilities. This avoids hard cascading errors.

### 1C. One-vs-Rest (OvR) Decomposition

Train three independent binary classifiers:
- Classifier A: `vol_churn` vs rest
- Classifier B: `invol_churn` vs rest
- Classifier C: `not_churned` vs rest

Final prediction: argmax of the three probability outputs (after calibration).

**Advantages:** Each classifier can have its own hyperparameters and feature subsets tailored to its task. Classifier B (invol) can lean heavily on billing/transaction features; Classifier A (vol) can lean on engagement/quiz features.  
**Disadvantages:** Probabilities from separate classifiers are not inherently calibrated relative to each other. Needs post-hoc calibration (Platt scaling or isotonic regression on each).

### 1D. One-vs-One (OvO)

Train three pairwise classifiers: vol-vs-invol, vol-vs-not, invol-vs-not. Aggregate via voting. Less common for this problem size but worth noting as a comparison point.

### Recommendation

Test **1A (flat multiclass)** and **1B (hierarchical two-stage)** as the primary framings. Use 1A as baseline. 1B is the higher-ceiling approach because the vol/invol distinction is domain-driven and benefits from specialized models.

---

## 2. Evaluation Strategy

### 2.1 Metrics

The competition mentions **Accuracy, Precision-Recall AUC, and F1-score**. For a 3-class problem:

- **Macro F1-score**: Unweighted average of per-class F1. Treats all three classes equally. This should be the **primary optimization target** since both churn types are business-critical despite being minorities.
- **Weighted F1-score**: Weighted by class support. Useful as secondary metric.
- **Per-class Precision & Recall**: Essential for the explainability component — judges will want to see that invol_churn recall is high (these are fixable) and vol_churn precision is high (don't waste retention budget on wrong users).
- **PR-AUC (macro)**: Average of per-class Precision-Recall AUC. More informative than ROC-AUC for imbalanced classes.
- **Log-loss (multi-class)**: Measures probability calibration quality. Important because the deliverable includes "probability scores for churn."
- **Confusion matrix**: The 3x3 matrix reveals specific misclassification patterns (e.g., is the model confusing vol with invol? Or is it failing to detect churn at all?).

### 2.2 Validation Scheme

- **Stratified K-Fold (K=5)**: Preserve the 50/25/25 class ratio in each fold. This is mandatory — random splits could produce folds with very different class distributions.
- **Repeated Stratified K-Fold (5×3)**: For final model comparison, repeat the 5-fold process 3 times with different random seeds to get confidence intervals on performance.
- **Hold-out set (15%)**: Before K-Fold, carve out a stratified 15% "local test set" that is **never** used during training or hyperparameter tuning. This is your ground truth for final ensemble selection and probability calibration. Since the competition test set has no labels, this is your only honest estimate.
- **Do NOT use the test set for anything** except final submission generation.

---

## 3. Class Imbalance Handling

The distribution is 50/25/25, so the imbalance is **moderate** — not extreme. Several approaches, ranked by recommended priority:

### 3A. Class Weights (Built-in)

All gradient boosting libraries (XGBoost, LightGBM, CatBoost) support `class_weight` or `scale_pos_weight`. For multi-class:
- XGBoost: Use custom sample weights via `sample_weight` array
- LightGBM: `class_weight='balanced'` or explicit dictionary `{0: 1, 1: 2, 2: 2}`
- CatBoost: `class_weights` or `auto_class_weights='Balanced'` / `'SqrtBalanced'`

**This should be the default first approach** — it's zero-cost, doesn't create synthetic data, and works well for moderate imbalance. The `SqrtBalanced` option in CatBoost is particularly interesting as it provides a softer rebalancing than full inverse-frequency weighting.

### 3B. SMOTE and Variants (Oversampling)

Generate synthetic minority samples. For multiclass, SMOTE applies per-class.

| Variant | Mechanism | When to use |
|---------|-----------|-------------|
| **SMOTE** | Interpolates between minority neighbors | Baseline oversampling |
| **Borderline-SMOTE** | Only oversamples near decision boundary | When classes have clear but tight boundaries |
| **ADASYN** | Density-adaptive — more synthesis in sparse minority regions | When minority has varied density |
| **SMOTE-ENN** | SMOTE + remove misclassified examples from majority | When majority has noise near boundary |
| **SMOTE-Tomek** | SMOTE + remove Tomek links (ambiguous pairs) | Cleaner version of SMOTE |

**Caution:** SMOTE on 90K samples with 100+ features can create overfitting. Apply SMOTE **only inside CV folds** (never before the split). Use `imblearn.pipeline.Pipeline` to chain SMOTE with the classifier.

**Recommended test:** Compare `class_weights='balanced'` vs SMOTE vs SMOTE-ENN. In churn literature, class weights often match or beat SMOTE for gradient boosting models. SMOTE shows more benefit for simpler models (Logistic Regression, SVM).

### 3C. Undersampling

Random undersampling of `not_churned` from 45K to ~22.5K to match each churn class. Fast but throws away data. Use as ablation study only.

### 3D. Focal Loss

A loss function that down-weights easy examples and focuses on hard-to-classify ones. Parametrized by γ (focusing parameter). Implemented in XGBoost via custom objective. Focal loss is particularly effective when the model is confident on the majority class but struggles with minority.

- `γ = 0`: equivalent to cross-entropy
- `γ = 1`: moderate focusing
- `γ = 2`: strong focusing (common default)
- `γ = 5`: aggressive — only very hard examples matter

**Implementation:** Custom objective function for XGBoost/LightGBM. CatBoost has `Focal` loss built-in via `loss_function='MultiClassOneVsAll:use_weights=true'`.

### 3E. Cost-Sensitive Learning

Assign different misclassification costs to different class pairs. For this problem:
- Misclassifying `invol_churn` as `not_churned` is expensive (we lose a fixable customer)
- Misclassifying `vol_churn` as `not_churned` is also expensive (we miss intervention window)
- Misclassifying `not_churned` as churn is mildly costly (wasted retention budget)
- Misclassifying `vol_churn` as `invol_churn` is moderately bad (wrong intervention type)

The cost matrix can be encoded as sample weights in training.

---

## 4. Feature Selection & Preprocessing

### 4.1 Feature Preprocessing

For tree-based models (our primary approach), minimal preprocessing is needed:
- **No scaling needed**: Trees are invariant to monotonic transformations
- **Missing value handling**: XGBoost/LightGBM/CatBoost handle NaN natively. Fill with 0 only for features where 0 is semantically meaningful (e.g., "no purchases" → 0 purchases)
- **Categorical encoding**: CatBoost handles categoricals natively. For XGBoost/LightGBM, one-hot encode low-cardinality categoricals (already done in feature engineering), use target encoding for high-cardinality ones

For linear/neural models (if used in ensemble):
- StandardScaler or RobustScaler (less sensitive to outliers)
- Log-transform heavily skewed features (total_spend, total_credits_spent)

### 4.2 Feature Selection Methods

With ~110-140 features (once generations are added), feature selection can improve generalization:

**Method 1: Null Importance (Permutation-Based)**
- Train model with real target, record feature importances
- Shuffle target 5-10 times, retrain, record importances
- Keep features whose real importance exceeds 95th percentile of null importances
- This eliminates features that only correlate by chance

**Method 2: Boruta**
- Creates "shadow features" (shuffled copies of all features)
- Iteratively tests whether real features outperform their shadow copies
- More principled than simple importance thresholding
- Use BorutaPy library with LightGBM as the estimator (fast)

**Method 3: Recursive Feature Elimination (RFE)**
- Train model, remove least important feature, repeat
- Computationally expensive but thorough
- Use RFECV (with cross-validation) to find optimal feature count

**Method 4: Correlation + VIF Filtering**
- Remove one of any feature pair with Pearson |r| > 0.95 (redundant)
- Compute Variance Inflation Factor (VIF), remove features with VIF > 10 (multicollinear)
- Apply before model-based selection

**Method 5: L1 Regularization Screening**
- Fit Logistic Regression with L1 penalty (high C → less regularization)
- Features with zero coefficients are candidates for removal
- Fast screening to reduce from 140 → ~80 features before heavier methods

**Recommended pipeline:** VIF filtering → Boruta → final model training. Or skip selection entirely for tree models (they handle irrelevant features via built-in importance weighting) and use it only for the stacking meta-learner.

---

## 5. Individual Models & Hyperparameter Spaces

### 5.1 LightGBM

The fastest of the three boosters; best for rapid iteration. Leaf-wise tree growth often yields better accuracy than level-wise (XGBoost default). Particularly strong on larger datasets.

**Hyperparameter search space:**

| Parameter | Range | Notes |
|-----------|-------|-------|
| `n_estimators` | 300–3000 | Higher is usually better with early stopping |
| `learning_rate` | 0.01–0.3 | Lower = more trees needed, but often better |
| `num_leaves` | 15–255 | Controls tree complexity. Higher → more complex |
| `max_depth` | 4–12, or -1 (unlimited) | Constrains tree depth. Use with num_leaves |
| `min_child_samples` | 5–100 | Minimum data in leaf. Regularization lever |
| `subsample` (bagging_fraction) | 0.5–1.0 | Row sampling per tree |
| `colsample_bytree` (feature_fraction) | 0.3–1.0 | Column sampling per tree |
| `reg_alpha` (L1) | 0–10 | L1 regularization on leaf weights |
| `reg_lambda` (L2) | 0–10 | L2 regularization on leaf weights |
| `min_split_gain` | 0–1.0 | Minimum gain to split. Pruning control |
| `class_weight` | `'balanced'` or dict | Imbalance handling |
| `boosting_type` | `'gbdt'`, `'dart'`, `'goss'` | dart often better but slower |

**Key tuning insights from literature:**
- Start with `learning_rate=0.05, n_estimators=1000, early_stopping_rounds=50`
- `num_leaves` is the most impactful single parameter — sweep it first
- For this dataset (90K rows, ~120 features), `num_leaves=63–127` is a reasonable starting point
- `dart` (Dropouts meet Multiple Additive Regression Trees) adds dropout to boosting rounds — reduces overfitting but 3-5x slower. Worth testing on final model
- `goss` (Gradient-based One-Side Sampling) speeds training by keeping only high-gradient samples. Useful if training time is tight

### 5.2 XGBoost

The most established booster. Strong regularization options. Level-wise tree growth is more conservative than LightGBM's leaf-wise.

**Hyperparameter search space:**

| Parameter | Range | Notes |
|-----------|-------|-------|
| `n_estimators` | 300–3000 | With early stopping |
| `learning_rate` (eta) | 0.01–0.3 | |
| `max_depth` | 3–10 | XGBoost primary complexity control |
| `min_child_weight` | 1–20 | Sum of instance weight in child. Higher = more conservative |
| `subsample` | 0.5–1.0 | |
| `colsample_bytree` | 0.3–1.0 | |
| `colsample_bylevel` | 0.3–1.0 | Additional column sampling per tree level |
| `gamma` | 0–5 | Minimum loss reduction for split |
| `reg_alpha` | 0–10 | L1 on weights |
| `reg_lambda` | 1–10 | L2 on weights (default=1, rarely set to 0) |
| `max_delta_step` | 0–10 | Useful for imbalanced data, caps leaf weight update |
| `tree_method` | `'hist'` | Always use histogram method for speed |
| `grow_policy` | `'depthwise'`, `'lossguide'` | lossguide = leaf-wise like LightGBM |

**Key insights:**
- `grow_policy='lossguide'` with `max_leaves` makes XGBoost behave like LightGBM
- `max_delta_step=1-5` specifically helps with imbalanced multiclass — it prevents the model from making overly aggressive updates on majority class samples
- XGBoost's `colsample_bylevel` (in addition to `bytree`) provides finer-grained feature sampling that can reduce overfitting
- For multiclass, use `objective='multi:softprob'` to get probability outputs (needed for calibration and ensembling)

### 5.3 CatBoost

Handles categorical features natively via ordered target statistics. Symmetric trees provide natural regularization. Slower to train but often competitive out-of-the-box.

**Hyperparameter search space:**

| Parameter | Range | Notes |
|-----------|-------|-------|
| `iterations` | 500–5000 | = n_estimators |
| `learning_rate` | 0.01–0.3 | |
| `depth` | 4–10 | CatBoost builds symmetric trees, so depth is the key complexity control |
| `l2_leaf_reg` | 1–30 | L2 on leaf values. CatBoost's primary regularization |
| `border_count` | 32–254 | Number of split candidates. Higher = finer but slower |
| `random_strength` | 0–10 | Randomization of split scores. Regularization |
| `bagging_temperature` | 0–5 | Bayesian bootstrap weight randomization |
| `min_data_in_leaf` | 1–50 | |
| `grow_policy` | `'SymmetricTree'`, `'Depthwise'`, `'Lossguide'` | Default Symmetric is usually best |
| `auto_class_weights` | `'Balanced'`, `'SqrtBalanced'` | Imbalance handling |
| `boosting_type` | `'Ordered'`, `'Plain'` | Ordered reduces prediction shift (overfitting) |

**Key insights:**
- CatBoost's `Ordered` boosting is specifically designed to prevent overfitting on small/medium datasets — it's worth keeping this as default
- `SqrtBalanced` class weights often outperform full `Balanced` — it provides softer correction
- `random_strength` is CatBoost-specific and acts as a regularizer by adding noise to split scores — values of 1-5 help prevent overfitting
- `bagging_temperature` controls the randomness of Bayesian bootstrap — higher values = more randomization = more regularization
- CatBoost is particularly good if we decide to keep some features as native categoricals (e.g., country_code, source, frustration) rather than one-hot encoding them

### 5.4 Random Forest

Bagging-based ensemble of decorrelated trees. Different bias-variance profile than boosting. Useful as a diversity source in stacking.

**Hyperparameter space:**

| Parameter | Range | Notes |
|-----------|-------|-------|
| `n_estimators` | 300–1500 | More is usually better; diminishing returns past ~500 |
| `max_depth` | 10–None | Deep trees = low bias, high variance |
| `min_samples_split` | 2–50 | |
| `min_samples_leaf` | 1–30 | |
| `max_features` | `'sqrt'`, `'log2'`, 0.3–0.8 | Feature subsetting. Key diversity lever |
| `class_weight` | `'balanced'`, `'balanced_subsample'` | |
| `max_samples` | 0.5–1.0 | Bootstrap sample size fraction |

**Why include:** RF provides decorrelated predictions from boosting models. In stacking, having both boosted and bagged base learners improves meta-learner generalization. Also fast to train and provides robust out-of-bag (OOB) error estimates.

### 5.5 Logistic Regression (Regularized)

A linear model. Low variance, high bias. Useful as:
- A baseline to sanity-check the data pipeline
- A meta-learner in stacking ensembles
- A calibration tool (Platt scaling)

**Hyperparameter space:**

| Parameter | Range | Notes |
|-----------|-------|-------|
| `penalty` | `'l1'`, `'l2'`, `'elasticnet'` | Regularization type |
| `C` | 0.001–100 (log-scale) | Inverse regularization strength |
| `solver` | `'saga'` (for elasticnet/l1), `'lbfgs'` (for l2) | |
| `l1_ratio` | 0–1 (for elasticnet) | Mix of L1 and L2 |
| `class_weight` | `'balanced'` | |
| `max_iter` | 1000–5000 | Convergence |

**Preprocessing required:** StandardScaler or RobustScaler mandatory. Log-transform skewed features.

### 5.6 Extra Trees (Extremely Randomized Trees)

Like Random Forest but with randomized split thresholds (not optimized). More variance, less overfitting risk. Another diversity source for stacking.

### 5.7 Neural Network (MLP)

A multi-layer perceptron for tabular data. Not typically competitive with boosted trees on structured data, but adds diversity to ensembles.

**Architecture to test:**
- Input → 256 → ReLU → Dropout(0.3) → 128 → ReLU → Dropout(0.3) → 64 → ReLU → 3 (softmax)
- Optimizer: Adam (lr=1e-3 with cosine decay)
- Batch size: 256–512
- Early stopping on validation loss
- Label smoothing (0.05–0.1) as regularization

**Why include:** Provides a fundamentally different inductive bias from tree models. If the true decision boundary has complex interactions that trees struggle with, the MLP might capture them. More valuable as an ensemble member than as a standalone model.

### 5.8 TabNet

Neural architecture specifically designed for tabular data. Uses sequential attention to select features at each step. Provides built-in feature importance.

**When to try:** If time permits after the core gradient boosting pipeline is optimized. TabNet's built-in feature selection can be informative for the explainability deliverable.

---

## 6. Ensemble & Stacking Architectures

Ensembling is where hackathon scores are won. The key insight from competition literature: **diversity matters more than individual model strength**. An ensemble of 5 mediocre-but-different models often beats the single best model.

### 6A. Simple Averaging / Soft Voting

Average the predicted probabilities from multiple models:

```
P(class=k) = (1/M) Σ P_m(class=k)
```

where M is the number of models.

**Weighted variant:** Optimize weights on the holdout set:
```
P(class=k) = Σ w_m * P_m(class=k),  where Σ w_m = 1
```

Use `scipy.optimize.minimize` with log-loss on the holdout set as the objective. Constrain weights to sum to 1 and be non-negative.

**Recommended first ensemble:** Weighted average of 3 tuned gradient boosters (LightGBM + XGBoost + CatBoost). This is the **default competition ensemble** and is hard to beat.

### 6B. Stacking (Level 2 Meta-Learning)

**Architecture:**

**Level 1 (Base Learners):** 3-5 diverse models trained via K-Fold CV. Each model produces out-of-fold (OOF) predictions on the training set.

**Level 2 (Meta-Learner):** A model trained on the Level 1 OOF predictions. Takes as input the probability vectors from each base learner (3 classes × M models = 3M features) and outputs the final 3-class prediction.

**Critical implementation detail:** Level 1 OOF predictions MUST be generated via cross-validation to prevent information leakage. For each fold k:
1. Train base learner on folds ≠ k
2. Predict on fold k → these are the OOF predictions for fold k's samples
3. Predict on the test set → average across all K models for test predictions

**Meta-learner options (ordered by recommendation):**

1. **Logistic Regression (L2):** Simple, fast, low overfitting risk. With only 3M ≈ 15-21 meta-features, a linear model is often sufficient. This is the most common and safest choice.
2. **Logistic Regression (ElasticNet):** L1 component can zero out uninformative base learner contributions.
3. **LightGBM (shallow):** max_depth=2-3, few trees. Can capture nonlinear interactions between base learners but risks overfitting.
4. **Ridge Classifier:** Like LogReg but uses squared loss. Sometimes faster and more stable.
5. **Neural Network (small):** 1 hidden layer, 16-32 units. Only if you have many base learners (5+).

**Recommended stacking ensemble:**

Level 1 base learners:
- LightGBM (tuned with Optuna)
- XGBoost (tuned with Optuna)
- CatBoost (tuned with Optuna)
- Random Forest (tuned)
- Extra Trees (tuned)

Level 2 meta-learner:
- Logistic Regression (L2, C tuned on inner CV)

**Variant — Stacking with original features:**
Concatenate the Level 1 OOF predictions with a subset of the original features (top 20 by importance). This gives the meta-learner access to both the base learner "opinions" and the raw signal. Can improve performance but increases overfitting risk — use sparingly and with strong regularization on the meta-learner.

### 6C. Blending

Simpler than stacking: use a single holdout split instead of K-Fold OOF predictions.

1. Split training data into 70% train / 30% blend
2. Train base learners on the 70%
3. Predict on the 30% → these are the blending features
4. Train meta-learner on the 30% with these features
5. For test predictions: predict with base learners on test, then with meta-learner

**Advantages:** Faster, less code, less risk of subtle leakage bugs.  
**Disadvantages:** Wastes 30% of training data for the base learners. The meta-learner sees fewer samples.

### 6D. Multi-Layer Stacking (3+ Levels)

Level 1 → Level 2 → Level 3. Each level uses OOF predictions from the prior.

**Use with extreme caution.** Beyond 2 levels, the marginal gain is tiny and overfitting risk is high. Only justified if you have many diverse base learners (8+) and a large training set. For 90K samples, stick to 2 levels max.

### 6E. Hill Climbing Ensemble Selection

Greedy forward selection: start with the best single model, iteratively add the model that improves the ensemble the most (evaluated on holdout). Stop when no model improves performance.

This is AutoGluon's approach and can be very effective at finding the optimal subset of models to include, since not all models improve the ensemble.

### 6F. Seed Averaging (Cheap Ensemble)

Train the same model architecture with different random seeds (5-10 seeds). Average predictions. This reduces variance at near-zero engineering cost and should be done for every final model.

---

## 7. Hierarchical / Two-Stage Decomposition

This section details the implementation of Approach 1B from Section 1.

### Stage 1: Churn vs Not-Churn (Binary)

**Target:** `is_churn = 1 if churn_status in {vol_churn, invol_churn} else 0`  
**Distribution:** 50/50 (perfectly balanced)  
**Model:** Best single gradient booster (likely LightGBM or XGBoost)  
**Expected performance:** High, since the combined churn signal is strong

### Stage 2: Voluntary vs Involuntary (Binary)

**Target:** `is_voluntary = 1 if vol_churn else 0` (trained only on churn samples)  
**Distribution:** 50/50 (balanced)  
**Training data:** Only the ~45K churn samples from training  
**Model:** Separate gradient booster with potentially different hyperparameters and feature emphasis

**Key insight:** Stage 2 can use a **different feature subset**. Specifically:
- Billing/transaction features should dominate for invol detection (failure rate, card issues, consecutive failures, prepaid/virtual cards)
- Engagement/quiz features should dominate for vol detection (generation frequency, recency, cost frustration, engagement decay)
- Allowing the Stage 2 model to focus on these differentiating features (via feature selection or higher importance weighting) should improve separation

### Final Probability Computation

```
P(not_churned) = P_stage1(not_churn)
P(vol_churn)   = P_stage1(churn) × P_stage2(voluntary)
P(invol_churn) = P_stage1(churn) × P_stage2(involuntary)
```

These three probabilities sum to 1 by construction.

### Hybrid Approach

Train both the flat multiclass model AND the hierarchical model. Then ensemble their probability outputs at the meta-level. This captures both "global" patterns (flat model) and "specialized" patterns (hierarchical model).

---

## 8. Explainability Layer

The competition weights **Actionable Insights at 40%** — equal to predictive performance. This means explainability isn't optional; it's half the score.

### 8.1 SHAP (SHapley Additive exPlanations)

The gold standard for model interpretability with tree models. TreeSHAP is exact and fast for gradient boosting.

**Global explanations:**
- SHAP summary plot: shows which features matter most across all users
- SHAP dependence plots: shows how a single feature's value affects prediction
- Group by churn type: separate SHAP analysis for vol_churn vs invol_churn to show the distinct drivers

**Local explanations (per-user):**
- SHAP waterfall plot: shows the contribution of each feature to a specific user's prediction
- This directly answers "why was this user flagged?" — the requirement from the case description

**Implementation:** Use `shap.TreeExplainer` for the gradient boosting models. For the ensemble, explain the meta-learner's decisions, which trace back to base learner predictions.

### 8.2 LIME (Local Interpretable Model-agnostic Explanations)

Complements SHAP for individual predictions. Creates a local linear approximation around each prediction. Useful for non-tree models in the ensemble.

### 8.3 Feature Importance Ranking by Churn Type

Run three separate importance analyses:
1. Features that drive `vol_churn` (expect: engagement decay, cost frustration, days since last generation, quiz incompleteness)
2. Features that drive `invol_churn` (expect: failure rate, max fail streak, prepaid card, country mismatch, last txn failed)
3. Features that protect against churn (expect: upsell purchase, high plan tier, high generation frequency, credit purchases)

This mapping directly feeds the **Strategy Proposal** deliverable.

### 8.4 Probability Calibration

For the probability scores to be trustworthy (and actionable), they must be calibrated:
- Use **Platt scaling** (sigmoid) or **isotonic regression** on the holdout set
- Verify with reliability diagrams (calibration curves)
- Well-calibrated probabilities mean: if we say "this user has 80% churn risk," then ~80% of similar users actually churn

---

## 9. Pipelines to Test (Priority-Ordered)

### Pipeline 1: Baseline — Single LightGBM (PRIORITY: HIGHEST)
- **Goal:** Establish baseline performance fast
- **Model:** LightGBM, `objective='multiclass'`, `class_weight='balanced'`
- **Tuning:** Manual grid search on 5 key params (learning_rate, num_leaves, min_child_samples, subsample, colsample_bytree)
- **Expected time:** 30 minutes
- **Why first:** Fast to train, strong out-of-box performance, reveals data quality issues

### Pipeline 2: Optuna-Tuned Big Three (PRIORITY: HIGHEST)
- **Goal:** Get best individual performance from each booster
- **Models:** LightGBM, XGBoost, CatBoost — each tuned independently with Optuna (50-100 trials each, TPE sampler, median pruner)
- **Imbalance:** `class_weight='balanced'` for all
- **Tuning space:** Full spaces from Section 5
- **Expected time:** 1-2 hours
- **Output:** Three tuned models + their OOF predictions (needed for ensembling)

### Pipeline 3: Weighted Average Ensemble (PRIORITY: HIGH)
- **Goal:** Quick ensemble gain
- **Method:** Optimize weights on holdout set using log-loss minimization
- **Inputs:** OOF predictions from Pipeline 2's three models
- **Expected improvement over best single:** 0.5-1.5% F1
- **Expected time:** 15 minutes

### Pipeline 4: Stacking Ensemble (PRIORITY: HIGH)
- **Goal:** Maximum predictive performance
- **Level 1:** LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees (5 base learners)
- **Level 2:** Logistic Regression (L2, C tuned)
- **OOF generation:** 5-fold stratified CV
- **Expected improvement over weighted average:** 0.5-1.0% F1
- **Expected time:** 1 hour

### Pipeline 5: Hierarchical Two-Stage (PRIORITY: HIGH)
- **Goal:** Exploit the domain structure of vol/invol distinction
- **Stage 1:** Optuna-tuned LightGBM (binary: churn vs not_churn)
- **Stage 2:** Optuna-tuned LightGBM (binary: vol vs invol, trained on churn samples only)
- **Final:** Multiply probabilities as described in Section 7
- **Expected time:** 1 hour

### Pipeline 6: Hierarchical + Flat Hybrid (PRIORITY: MEDIUM)
- **Goal:** Combine global and specialized perspectives
- **Method:** Weighted average of Pipeline 4's output and Pipeline 5's output
- **Weight optimization:** On holdout set
- **Expected time:** 15 minutes

### Pipeline 7: Imbalance Ablation (PRIORITY: MEDIUM)
- **Goal:** Find the best imbalance handling strategy
- **Test matrix** (on best single model from Pipeline 2):
  - No rebalancing (raw)
  - `class_weight='balanced'`
  - `class_weight='SqrtBalanced'` (CatBoost only)
  - SMOTE (k_neighbors=5)
  - SMOTE-ENN
  - ADASYN
  - Focal loss (γ ∈ {1, 2, 5})
- **Evaluate:** Macro F1 on stratified 5-fold CV
- **Expected time:** 1 hour

### Pipeline 8: Feature Selection Ablation (PRIORITY: MEDIUM)
- **Goal:** Determine if trimming features helps
- **Test:**
  - All features (baseline)
  - Top 80 by Boruta
  - Top 50 by permutation importance
  - Remove VIF > 10 features
- **On:** Best single model from Pipeline 2
- **Expected time:** 45 minutes

### Pipeline 9: Seed Averaging (PRIORITY: MEDIUM)
- **Goal:** Reduce variance of final model
- **Method:** Train final ensemble with 10 different random seeds, average predictions
- **Expected improvement:** 0.1-0.3% F1 (small but free)
- **Expected time:** Scale of Pipeline 4 × number of seeds

### Pipeline 10: Neural Meta-Learner (PRIORITY: LOW)
- **Goal:** Test if nonlinear meta-learning helps
- **Method:** Replace Pipeline 4's LogReg meta-learner with a small MLP (32→16→3)
- **Risk:** High overfitting risk with only 5×3=15 meta-features
- **Expected time:** 30 minutes

### Pipeline 11: TabNet + Diversity (PRIORITY: LOW)
- **Goal:** Add a fundamentally different model to the ensemble
- **Method:** Train TabNet, add its OOF predictions to Pipeline 4's stacking
- **Expected time:** 1 hour (TabNet is slow)

### Pipeline 12: One-vs-Rest Specialists (PRIORITY: LOW)
- **Goal:** Test 3 specialized binary classifiers
- **Method:** Three LightGBMs, each tuned for one class, with Platt scaling calibration, argmax for final prediction
- **Expected time:** 1 hour

---

## 10. Experimental Protocol

### 10.1 Hyperparameter Tuning with Optuna

```
# Pseudocode for Optuna objective
def objective(trial):
    params = {
        'num_leaves': trial.suggest_int('num_leaves', 15, 255),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'min_child_samples': trial.suggest_int('min_child', 5, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample', 0.3, 1.0),
        'reg_alpha': trial.suggest_float('alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('lambda', 1e-8, 10.0, log=True),
    }
    
    scores = []
    for fold in stratified_kfold(5):
        model = LGBMClassifier(**params, early_stopping_rounds=50)
        model.fit(X_train_fold, y_train_fold, eval_set=[(X_val_fold, y_val_fold)])
        pred = model.predict_proba(X_val_fold)
        scores.append(macro_f1(y_val_fold, pred.argmax(1)))
    
    return np.mean(scores)

study = optuna.create_study(direction='maximize', sampler=TPESampler(), pruner=MedianPruner())
study.optimize(objective, n_trials=100)
```

**Optuna configuration:**
- Sampler: TPE (Tree-structured Parzen Estimator) — default, well-suited for this search space
- Pruner: MedianPruner (early-stop unpromising trials at fold 2/5) or HyperbandPruner
- n_trials: 50 for quick iteration, 100 for final tuning
- `study.best_params` → retrain on full training data with early stopping on holdout

### 10.2 Tracking & Comparison

For each pipeline, record:
- Macro F1 (mean ± std across folds)
- Per-class F1 (not_churned, vol_churn, invol_churn)
- PR-AUC (macro)
- Log-loss
- Training time
- Number of features used
- Confusion matrix on holdout set

### 10.3 Final Submission Strategy

1. Select the best pipeline based on holdout performance
2. Apply seed averaging (10 seeds)
3. Calibrate probabilities (isotonic regression on holdout)
4. Generate test predictions
5. Extract SHAP explanations for top 50 highest-risk users
6. Compile the strategy proposal based on SHAP feature clusters

### 10.4 Time Budget Allocation (Assuming 8-Hour Hackathon Sprint)

| Phase | Time | Pipelines |
|-------|------|-----------|
| Data loading, feature verification | 30 min | — |
| Pipeline 1 (Baseline) | 30 min | P1 |
| Pipeline 2 (Optuna tuning) | 2 hours | P2 |
| Pipeline 3 + 4 (Ensembles) | 1.5 hours | P3, P4 |
| Pipeline 5 + 6 (Hierarchical) | 1 hour | P5, P6 |
| Pipeline 7 (Imbalance ablation) | 45 min | P7 |
| SHAP + Explainability | 45 min | — |
| Presentation prep | 1 hour | — |

Pipelines 8-12 are stretch goals if ahead of schedule.

### Pipeline 13: Synthetic Data Artifact Exploitation — KGMON Playbook (PRIORITY: HIGH)

This pipeline is inspired by the **1st place solution to Kaggle Playground Series S6E3** (the "KGMON Playbook"), a churn prediction competition on CTGAN-generated synthetic data where the winning approach systematically reverse-engineered the generator's fingerprints and used them as features. Our Higgsfield hackathon data is **confirmed synthetic** (year 1067, clearly obfuscated dates) and shows strong generator artifacts (massive Benford's Law violation, tight value clustering around 33 anchor price points, non-uniform decimal digit distributions). This makes the KGMON techniques directly applicable.

#### 13.1 Core Concept: The Generator Is a Feature

When synthetic data is produced by CTGAN, TVAE, or any deep generative model from an original dataset, the generation process introduces systematic, exploitable artifacts:

1. **Value clustering:** Synthetic continuous values cluster around the original dataset's true values. The distance from each synthetic value to its nearest "original" value is a feature.
2. **Decimal digit patterns:** The generator's sampling/rounding behavior leaves non-uniform digit distributions at each decimal position. These digit distributions differ between churn classes because the generator conditions on the target.
3. **Frequency drift:** Some original values get oversampled by the generator while others get undersampled. The ratio `synthetic_count / original_count` for each value encodes generator behavior that correlates with the target.
4. **Rounding artifacts:** Integer-like values, quarter-like values, and half-like values appear at rates inconsistent with real financial data. Counting how many of a row's features look "too round" is a signal.

#### 13.2 Evidence in Our Data

Analysis of the Higgsfield test set reveals strong synthetic fingerprints:

**Benford's Law violation (amount_in_usd):**
| Leading digit | Actual | Benford Expected | Delta |
|:---:|:---:|:---:|:---:|
| 1 | 0.120 | 0.301 | **−0.181** |
| 4 | 0.246 | 0.097 | **+0.149** |
| 9 | 0.173 | 0.046 | **+0.127** |

This is a massive deviation — real financial data follows Benford's Law closely. The distortion means the generator favored certain price ranges, and the pattern of distortion differs by churn class.

**Value clustering:** 33 "anchor" price points account for the vast majority of transaction amounts. The amounts 9.0, 29.0, 49.0 appear thousands of times (likely Higgsfield's actual subscription tiers), while non-round values like 59.29, 53.90, 58.31 appear at suspiciously regular frequencies (~150 each), suggesting they are generator variations around true pricing.

**Decimal digit non-uniformity:** First decimal digit d=8 appears 23.8% of the time (expected ~10% for uniform), d=4 appears only 2.4%. The generator has a digit preference that encodes information about the original data point being perturbed.

#### 13.3 Feature Engineering — Snap Features

**Step 1: Infer the original pricing grid.**

From the training data, identify the "anchor" values — amounts appearing with disproportionately high frequency. These are the generator's source values.

```python
# Identify anchor prices (appearing >50 times across train+test)
all_amounts = pd.concat([train_txn['amount_in_usd'], test_txn['amount_in_usd']])
freq = all_amounts.value_counts()
anchor_prices = freq[freq > threshold].index.sort_values().values
# Result: array([5, 9, 9.9, 10, 10.71, ..., 249, 352.8]) — ~33 values
```

**Step 2: Snap each synthetic value to its nearest anchor.**

```python
from scipy.spatial import cKDTree

tree = cKDTree(anchor_prices.reshape(-1, 1))
distances, indices = tree.query(df['amount_in_usd'].values.reshape(-1, 1))

df['amount_snap']      = anchor_prices[indices.flatten()]  # recovered "true" value
df['amount_snap_diff'] = df['amount_in_usd'] - df['amount_snap']  # generator noise
df['amount_snap_dist'] = distances.flatten()  # absolute distance to anchor
```

The snap value recovers the original price point; the diff encodes the magnitude and direction of the generator's perturbation. Both are features.

**Apply to every numeric column with suspected synthetic generation:**
- `amount_in_usd` → snap to inferred pricing grid
- `purchase_amount_dollars` → snap to inferred pricing grid (same or similar anchors)
- `credit_cost` (generations table) → snap to inferred credit pricing tiers

#### 13.4 Feature Engineering — Decimal Digit Extraction

Extract digit-position features from every numeric column:

```python
def extract_digit_features(series, col_name):
    x = series.values
    frac = x - np.floor(x)
    features = {
        f'{col_name}_d1': np.floor(frac * 10).astype(int),          # 1st decimal
        f'{col_name}_d2': (np.floor(frac * 100) % 10).astype(int),  # 2nd decimal
        f'{col_name}_frac100': np.round(frac * 100).astype(int),    # 2-digit frac
        f'{col_name}_mod10': (np.floor(x) % 10).astype(int),        # last digit
        f'{col_name}_mod100': (np.floor(x) % 100).astype(int),      # last 2 digits
        f'{col_name}_is_round': (frac < 0.005).astype(int),         # round flag
        f'{col_name}_is_quarter': (np.abs(frac - 0.25) < 0.005 | 
                                    np.abs(frac - 0.5) < 0.005 |
                                    np.abs(frac - 0.75) < 0.005).astype(int),
    }
    return pd.DataFrame(features)
```

Apply to: `amount_in_usd`, `purchase_amount_dollars`, and any numeric column in the generations table (`credit_cost`, `duration`).

#### 13.5 Feature Engineering — Benford's Law Deviation

Compute per-row Benford deviation as a feature:

```python
def benford_deviation(x):
    """How much does this value's leading digit deviate from Benford expectation?"""
    if x <= 0: return 0
    d = int(str(abs(x)).lstrip('0').replace('.','')[0])
    benford_p = np.log10(1 + 1/d)
    return benford_p  # low = suspicious digit for this magnitude

# Per-row: average Benford probability across all numeric columns
df['benford_score'] = df[numeric_cols].apply(
    lambda row: np.mean([benford_deviation(v) for v in row if v > 0]), axis=1
)
```

#### 13.6 Feature Engineering — Frequency / Count Encoding on Snap Values

```python
# How many times does this snap value appear? (measures generator oversampling)
snap_freq = train_data['amount_snap'].value_counts(normalize=True)
df['snap_freq'] = df['amount_snap'].map(snap_freq)

# If original data is available: count ratio = synthetic_freq / original_freq
# This measures generator sampling bias per archetype
```

#### 13.7 Feature Engineering — Row-Level Artifact Aggregation

Aggregate artifact signals across all numeric columns in a row:

```python
# How many of this row's numeric values are suspiciously round?
df['n_round_values'] = sum((df[c] == df[c].round(0)).astype(int) for c in numeric_cols)

# How many are near a quarter/half?
df['n_quarter_values'] = sum(
    ((df[c] - df[c].round(0)).abs() - 0.25).abs() < 0.01
    for c in numeric_cols
)

# Average snap distance across all snappable columns
df['mean_snap_distance'] = df[[c for c in df.columns if 'snap_dist' in c]].mean(axis=1)
```

#### 13.8 Feature Engineering — KDTree Original Lookup

If any "original" reference data is available (e.g., Higgsfield provides a seed dataset, or we infer the grid), build a KDTree on the original data's standardized numeric columns and attach the nearest original row's churn label:

```python
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler

# Fit on inferred "original" rows (the anchor grid)
scaler = StandardScaler()
orig_features = scaler.fit_transform(original_data[numeric_cols])
tree = cKDTree(orig_features)

# Query each synthetic row
syn_features = scaler.transform(df[numeric_cols])
distances, indices = tree.query(syn_features, k=5)

df['nn_orig_dist_1'] = distances[:, 0]
df['nn_orig_dist_mean'] = distances.mean(axis=1)
# If original has labels: df['nn_orig_churn'] = original_data['churn'].iloc[indices[:, 0]].values
```

#### 13.9 Model Configuration

The KGMON solution showed that these artifact features benefit from **high bin counts** in tree models to capture fine-grained decimal patterns:

```python
# XGBoost: allow fine splits on decimal artifacts
xgb_params = {
    'max_bin': 16000,  # default 256; need high bins for digit features
    'tree_method': 'hist',
}

# LightGBM:
lgbm_params = {
    'max_bin': 8192,  # default 255
}
```

The digit and snap features should be treated as **categoricals** (integer-valued with bounded range), not continuous numerics.

#### 13.10 Full KGMON-Style Pipeline

**Phase A — Artifact Feature Engineering:**
1. Infer anchor price grid from train+test frequency analysis
2. Compute snap features for all numeric columns (snap_value, snap_diff, snap_dist)
3. Extract decimal digit features (d1, d2, frac100, mod10, mod100, is_round)
4. Compute Benford deviation scores
5. Frequency encoding on snap values
6. Row-level artifact aggregation (n_round, mean_snap_dist)

**Phase B — Augment Existing Feature Matrix:**
- Take the 111+ features from `feature_engineering.py`
- Append ~40-60 artifact features from Phase A
- Total: ~160-170 features

**Phase C — Train with High-Bin Configuration:**
- LightGBM with `max_bin=8192`, Optuna-tuned
- XGBoost with `max_bin=16000`, Optuna-tuned
- CatBoost with `border_count=254` (its maximum)
- Use the same 5-fold stratified CV as other pipelines

**Phase D — Stack into Main Ensemble:**
- Generate OOF predictions from Phase C models
- Add as Level 1 base learners to the Pipeline 4 stacking ensemble
- The artifact-aware models capture a fundamentally different signal (generator behavior) from the regular models (user behavior), providing genuine ensemble diversity

**Expected impact:** In the Kaggle PS-S6E3 competition, snap features and digit extraction were described as "the dominant driver of performance" and were used in "nearly every model" of the 150-model ensemble. For our data, the confirmed Benford's Law violations and value clustering suggest similar gains are possible.

**Expected time:** 1.5 hours (30 min feature engineering, 30 min model training, 30 min integration with ensemble)

#### 13.11 Applicability Assessment

**When this helps most:** If the hackathon organizers generated their synthetic data via CTGAN/TVAE from a real Higgsfield user dataset (likely, given the case description says "synthetic/anonymized"), then the generator's fingerprints carry signal about the original labels that the model can exploit.

**When this helps less:** If the data was synthesized via rule-based simulation (not deep generative), the artifacts will be less structured. However, even rule-based generators leave patterns (e.g., the 33 anchor prices are likely Higgsfield's real pricing tiers regardless of generation method).

**Risk:** If the hidden test set was generated with a different random seed or generator configuration, artifact features might not transfer. Mitigate by: using artifact features as *additions* to the core behavioral features (not replacements), and monitoring CV variance — if artifact features have high variance across folds, downweight them.

### Updated Time Budget (Revised)

| Phase | Time | Pipelines |
|-------|------|-----------|
| Data loading, feature verification | 30 min | — |
| Pipeline 1 (Baseline) | 30 min | P1 |
| Pipeline 2 (Optuna tuning) | 1.5 hours | P2 |
| **Pipeline 13 (Artifact features + models)** | **1.5 hours** | **P13** |
| Pipeline 3 + 4 (Ensembles, now including P13 models) | 1.5 hours | P3, P4 |
| Pipeline 5 + 6 (Hierarchical) | 1 hour | P5, P6 |
| SHAP + Explainability | 45 min | — |
| Presentation prep | 45 min | — |

Pipeline 13 is elevated to **HIGH priority** and slotted early because: (a) artifact features feed into all downstream ensembles, (b) the evidence of synthetic generation in our data is strong, and (c) it's the single highest-impact technique from the best-performing solution on a nearly identical problem (Kaggle churn on synthetic data).
