# CreditCard-FraudDetection
Built a recall-optimized, explainable fraud detection system using tree-based ensemble models with imbalance handling, robust validation, and production-ready packaging supporting both real-time and batch inference.

1️⃣ Problem Framing

A binary classification system to detect fraudulent credit card transactions, with recall optimization as the primary objective, since missing fraud (false negatives) has higher business cost than false positives.

The system supports:

Real-time single prediction (manual input)

Batch scoring via file upload

Explainability via SHAP

Adjustable fraud threshold based on business risk appetite

🧠 Data Engineering & Exploration
Data Quality & Structure

Validated data types

Checked for nulls and inconsistencies

Ensured classification framing

Verified class imbalance

Class Imbalance Handling

Fraud datasets are inherently imbalanced. We:

Measured fraud distribution

Used Stratified Train-Test Split to preserve fraud ratio

Applied:

SMOTE (synthetic minority oversampling)

ADASYN (adaptive synthetic sampling for hard-to-learn areas)

This ensured better minority class representation during training.

📊 Exploratory Data Analysis

Performed:

Univariate Analysis → distribution, skewness, outliers

Bi-variate Analysis → feature vs fraud behavior

Correlation Heatmap → detect redundancy and linear relationships

Removed unnecessary and redundant features to reduce noise.

⚙️ Feature Engineering

Created high-signal derived features:

Age (from DOB)

Transaction time components (year, month, day, hour)

Temporal grouping (year-month)

Applied:

One-hot encoding (low-cardinality)

Label encoding (high-cardinality features)

Final feature schema was fixed and versioned to ensure inference consistency.

🤖 Modeling Strategy

We evaluated multiple tree-based models due to their strength in tabular datasets:

1️⃣ Decision Tree

GridSearchCV

Small search space → exhaustive grid

Optimized for Recall

4-fold Cross Validation

2️⃣ Random Forest

Larger hyperparameter space

RandomizedSearchCV for efficiency

Handles variance & improves generalization

3️⃣ XGBoost

Boosting-based sequential error correction

Strong performance in imbalanced tabular datasets

Built-in regularization

Captures non-linear interactions

XGBoost performed best overall in balancing recall and generalization.

📈 Evaluation Strategy

Because of imbalance, we avoided relying on accuracy.

We evaluated using:

ROC-AUC (~0.87) → discrimination ability

Precision → control false positives

Recall → prioritize fraud capture

F1-score → balance precision & recall

Primary optimization metric: Recall

Business rationale:

Missing fraud is more costly than reviewing legitimate transactions.

🔎 Feature Selection Approach

No explicit wrapper/filter method used, but:

Tree-based models inherently perform implicit feature selection:

Only split on features reducing impurity

Ignore irrelevant variables

Capture non-linear interactions

Additionally, we manually removed clearly redundant features.

🏗 Productionization Layer

The system is packaged as:

Trained model

Encoders

Expected feature schema

All bundled into a serialized artifact.

Deployed using Streamlit interface with:

Adjustable fraud threshold

Batch scoring

SHAP explainability (Force + Waterfall plots)

Probability-based interpretation

🧩 Explainability & Transparency

We integrated SHAP:

Force plot (local explanation)

Waterfall plot (probability-adjusted contribution view)

This ensures:

Model transparency

Auditability

Regulatory defensibility

Stakeholder trust

🏆 Architectural Strengths

✔ Handles class imbalance rigorously
✔ Uses stratified validation
✔ Hyperparameter tuned
✔ Recall-optimized
✔ Model comparison framework
✔ Implicit feature selection via tree models
✔ Explainable AI integration
✔ Flexible threshold for risk appetite
✔ Supports real-time and batch inference

📊 Performance Snapshot

ROC-AUC: ~0.87

Recall optimized

Balanced F1 performance

Stable cross-validation behavior

🎯 Business Impact

This system enables:

Early fraud detection

Reduced financial losses

Controlled false positive rates

Operational flexibility via threshold tuning

Explainable decision support for fraud analysts
