# Employee Attrition Prediction

A machine learning project that predicts employee attrition using the IBM HR Analytics dataset. The project evaluates five classification algorithms, addresses class imbalance through SMOTE, and applies hyperparameter tuning to optimize model performance.

## Problem Statement

Employee attrition poses significant operational and financial challenges to organizations. This project builds predictive models to identify employees at risk of leaving, enabling proactive retention strategies. The dataset exhibits a class imbalance with an attrition rate of approximately 16.12%, making it a realistic and challenging classification problem.

## Dataset

- **Source:** IBM HR Analytics Employee Attrition & Performance
- **Records:** 1,470 employees
- **Features:** 35 attributes covering demographics, job characteristics, satisfaction metrics, and compensation
- **Target Variable:** Attrition (Yes/No)

Key features include Age, MonthlyIncome, OverTime, JobSatisfaction, YearsAtCompany, DistanceFromHome, WorkLifeBalance, and EnvironmentSatisfaction, among others.

## Project Structure

```
Employee-Attrition-Prediction/
├── Employee Attrition Prediction.ipynb   # Full analysis notebook
├── Dataset.csv                           # IBM HR dataset
├── .gitignore
└── README.md
```

## Methodology

### Phase 1: Baseline Model Development

1. **Data Exploration and Quality Assessment**
   - Descriptive statistics and distribution analysis
   - Missing value and duplicate checks
   - Identification of constant and near-constant columns

2. **Exploratory Data Analysis**
   - Correlation analysis to identify top features associated with attrition
   - Feature distribution comparison between attrition groups
   - Outlier detection using the IQR method
   - Multicollinearity assessment

3. **Data Preprocessing**
   - Label encoding for categorical variables (Gender, Department, BusinessTravel, MaritalStatus, EducationField, JobRole)
   - Binary encoding for the target variable
   - Removal of constant-value columns (EmployeeCount, Over18, StandardHours)
   - Feature scaling with StandardScaler
   - Stratified 80/20 train-test split

4. **Model Training and Evaluation**

   Five classification algorithms were trained and compared:

   | Model               | Accuracy | Precision | Recall | F1-Score |
   |---------------------|----------|-----------|--------|----------|
   | Logistic Regression | 0.7449   | 0.36      | 0.79   | 0.50     |
   | SVM (RBF Kernel)    | 0.8129   | 0.44      | 0.64   | 0.52     |
   | Random Forest       | 0.8401   | 0.50      | 0.09   | 0.15     |
   | Decision Tree       | 0.7687   | 0.35      | 0.51   | 0.41     |
   | XGBoost             | 0.8537   | 0.57      | 0.34   | 0.43     |

   **Selected Model:** Logistic Regression -- chosen for its highest recall (0.79), which is the priority metric for attrition prediction. Capturing at-risk employees is more valuable than overall accuracy in this business context.

### Phase 2: Improvement Strategies

1. **SMOTE (Synthetic Minority Oversampling Technique)**
   - Balanced training data from 986:190 to 986:986
   - Validated across all five algorithms to confirm model selection

2. **Hyperparameter Tuning**
   - GridSearchCV with StratifiedKFold cross-validation
   - Optimized regularization strength (C), penalty type, and solver
   - Decision threshold adjustment (0.47) for recall optimization

3. **Combined Strategy: SMOTE + Hyperparameter Tuning**
   - Achieved improved recall (0.77) with better precision-recall balance
   - F1-Score improved to 0.51 compared to tuning-only approaches

## Key Findings

- **OverTime**, **Age**, and **YearsAtCompany** are the strongest predictors of attrition
- Models optimized for accuracy tend to underperform on the minority class (attrition)
- Logistic Regression consistently delivers the best recall across all experimental scenarios
- SMOTE combined with hyperparameter tuning provides the most balanced performance
- Interpretable model coefficients allow HR teams to understand and act on attrition drivers

## Tech Stack

- **Language:** Python 3
- **Data Processing:** NumPy, Pandas
- **Visualization:** Matplotlib, Seaborn
- **Statistical Analysis:** SciPy
- **Machine Learning:** scikit-learn, XGBoost
- **Class Imbalance:** imbalanced-learn (SMOTE)

## How to Run

1. Ensure Python 3.x is installed with the required packages:
   ```
   pip install numpy pandas matplotlib seaborn scipy scikit-learn xgboost imbalanced-learn
   ```

2. Open the Jupyter notebook:
   ```
   jupyter notebook "Employee Attrition Prediction.ipynb"
   ```

3. Run all cells sequentially.

## Evaluation Metrics

All models are evaluated using:
- **Accuracy** -- Overall correctness
- **Precision** -- Of predicted attritions, how many were correct
- **Recall** -- Of actual attritions, how many were identified (priority metric)
- **F1-Score** -- Harmonic mean of precision and recall
- **ROC-AUC** -- Model's ability to discriminate between classes

## License

This project is intended for academic and educational purposes only.
