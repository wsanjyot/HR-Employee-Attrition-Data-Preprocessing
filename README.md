# HR Employee Attrition — Data Preprocessing

A structured machine learning data preprocessing project using the **IBM HR Employee Attrition dataset** and Scikit-learn.

This project focuses on transforming raw HR data into model-ready numerical features using feature selection, train-test splitting, scaling, categorical encoding, and `ColumnTransformer`.

---

## Project Overview

Machine learning datasets often contain a mixture of numerical, ordinal, categorical, constant, and identifier features.

This project demonstrates how to build a clean preprocessing workflow for such a dataset while avoiding data leakage.

### Dataset

**IBM HR Employee Attrition Dataset**

* Rows: 1,470
* Original columns: 35
* Target: `Attrition`

Target values:

```text
Yes → Employee left the organization
No  → Employee stayed
```

---

## Objectives

* Perform initial dataset inspection
* Identify feature types
* Remove irrelevant and constant features
* Separate features and target
* Split data into training and testing sets
* Scale numerical features
* Handle ordinal numerical features
* Encode categorical features
* Build a `ColumnTransformer`
* Prevent data leakage
* Generate processed feature names
* Convert processed data into pandas DataFrames

---

## Preprocessing Workflow

```text
Raw HR Dataset
      │
      ▼
Initial Data Inspection
      │
      ▼
Remove Irrelevant Columns
      │
      ▼
Separate Features and Target
      │
      ▼
Train-Test Split
      │
      ▼
Feature Classification
      │
      ├──────────────┬───────────────┐
      ▼              ▼               ▼
 Numerical        Ordinal        Categorical
      │              │               │
      ▼              ▼               ▼
StandardScaler  StandardScaler  OneHotEncoder
      │              │               │
      └──────────────┼───────────────┘
                     ▼
              ColumnTransformer
                     │
                     ▼
          Fit on Training Data
                     │
                     ▼
             Transform Data
                     │
                     ▼
            51 Processed Features
```

---

## Feature Classification

### Numerical Features

```python
numerical_cols = [
    "Age",
    "DailyRate",
    "DistanceFromHome",
    "HourlyRate",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]
```

### Ordinal Features

These are discrete numerical variables where the order of values carries meaning.

```python
ordinal_cols = [
    "Education",
    "EnvironmentSatisfaction",
    "JobInvolvement",
    "JobLevel",
    "JobSatisfaction",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "WorkLifeBalance"
]
```

### Categorical Features

```python
categorical_cols = [
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]
```

---

## Removed Columns

The following columns were removed:

```text
EmployeeCount
EmployeeNumber
Over18
StandardHours
```

### Reasons

| Column         | Reason           |
| -------------- | ---------------- |
| EmployeeCount  | Constant feature |
| EmployeeNumber | Identifier       |
| Over18         | Constant feature |
| StandardHours  | Constant feature |

These columns do not provide useful varying information for the preprocessing workflow.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* NumPy
* Jupyter Notebook / Google Colab

---

## Key Scikit-learn Components

### StandardScaler

Used for numerical and ordinal numerical features.

```python
StandardScaler()
```

### OneHotEncoder

Used for nominal categorical variables.

```python
OneHotEncoder(handle_unknown="ignore")
```

### ColumnTransformer

Used to apply different preprocessing operations to different feature groups.

```python
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("ord", StandardScaler(), ordinal_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)
```

---

## Train-Test Preprocessing

The preprocessing object is fitted only on training data:

```python
X_train_processed = preprocessor.fit_transform(X_train)
```

The test data is transformed using the already-fitted preprocessor:

```python
X_test_processed = preprocessor.transform(X_test)
```

This prevents information from the test dataset from influencing the preprocessing parameters.

---

## Final Feature Count

The processed dataset contains **51 features**.

```text
14 Numerical Features
+
9 Ordinal Features
+
28 One-Hot Encoded Features
=
51 Features
```

Categorical expansion:

```text
BusinessTravel       → 3
Department           → 3
EducationField       → 6
Gender               → 2
JobRole              → 9
MaritalStatus        → 3
OverTime             → 2

Total                → 28
```

---

## Final Dataset Shapes

Training data:

```text
(1029, 51)
```

Testing data:

```text
(441, 51)
```

---

## Project Structure

```text
HR-Employee-Attrition-Preprocessing/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── notebooks/
│   └── hr_employee_attrition_preprocessing.ipynb
│
├── processed_data/
│   ├── X_train_processed.csv
│   ├── X_test_processed.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository and install the required dependencies.

```bash
git clone <your-repository-url>
cd HR-Employee-Attrition-Preprocessing
```

Install dependencies:

```bash
pip install pandas numpy scikit-learn jupyter
```

Or:

```bash
pip install -r requirements.txt
```

---

## Learning Outcomes

Through this project, I practiced:

* Understanding feature semantics instead of relying only on datatypes
* Differentiating numerical, ordinal, and categorical variables
* Standardization using `StandardScaler`
* One-hot encoding using `OneHotEncoder`
* Building preprocessing pipelines with `ColumnTransformer`
* Understanding `fit_transform()` and `transform()`
* Preventing data leakage
* Extracting transformed feature names
* Converting transformed NumPy arrays into pandas DataFrames

---

## Next Step

The processed dataset provides the foundation for the next stage of the machine learning workflow.

Future work can include:

```text
Processed Data
      ↓
Model Training
      ↓
Prediction
      ↓
Evaluation
      ↓
Model Comparison
```

The current repository focuses specifically on **data preprocessing, scaling, and encoding**.
