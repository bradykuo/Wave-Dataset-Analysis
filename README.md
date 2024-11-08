# Wave Dataset Analysis 

## Overview
This project implements and compares different classification and variable selection methods on the modified wave dataset (Breiman et al., 1984; Rakotomalala, 2005). The analysis is implemented in both R and Python, providing comprehensive comparisons of different modeling approaches.

## Dataset Description
- **Size**: 33,334 subjects (10,000 for training)
- **Features**: 
  - 21 active variables (based on wave patterns)
  - 100 noise variables (independent from classification)
- **Response**: Binary classification (balanced between two classes)
- **Original Source**: Breiman et al., 1984, modified by Rakotomalala (2005)

## Implementations

### R Implementation
1. **LASSO Classification** (`LASSO.R`)
   - LASSO-based logistic regression
   - Cross-validation for parameter tuning
   - ROC curve analysis

2. **Grafting Method** (`Grafting.R`)
   - Chebychev's Greedy Algorithm implementation
   - Gradient-based variable selection

3. **Linear Regression Approaches** (`linear regression.R`)
   - Stepwise regression (AIC)
   - Ridge regression
   - Elastic Net regression

### Python Implementation (`wave_dataset_analysis.py`)
1. **Linear Models**
   - Logistic Regression (with L1 regularization)
   - Linear SVC
   - Linear Discriminant Analysis (LDA)

2. **Tree-based Models**
   - Random Forest
   - XGBoost

## Key Results
- Best performing model: Logistic Regression (92.15% accuracy)
- Model Performance Ranking:
  1. Logistic Regression
  2. Linear SVM
  3. Random Forest
  4. Linear Regression

## Dependencies

### R Dependencies
```R
library(glmnet)    # For regularization methods
library(pROC)      # For ROC curve analysis
library(caret)     # For data splitting
library(dplyr)     # For data manipulation
library(ggplot2)   # For visualization
library(gridExtra) # For plot arrangement
library(MASS)      # For stepwise regression
```

### Python Dependencies
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

## Usage

### R Analysis
```R
# LASSO Analysis
source("LASSO.R")
results <- analyze_wave_dataset()

# Grafting Analysis
source("Grafting.R")
grafting_results <- analyze_wave_dataset_grafting()

# Linear Regression Analysis
source("linear regression.R")
methods <- c("stepwise", "ridge", "elastic_net")
results_list <- lapply(methods, function(m) {
  analyze_wave_dataset(method = m)
})
```

### Python Analysis
```python
# Load and preprocess data
data1, meta1 = arff.loadarff(train_file)
data2, meta2 = arff.loadarff(test_file)

# Run different models
logistic_results = train_logistic_model(X_train, y_train)
svm_results = train_svm_model(X_train, y_train)
rf_results = train_random_forest(X_train, y_train)
```

## Key Findings
1. Feature importance analysis shows predictive variables have significantly higher importance than noise variables
2. Linear models (Logistic Regression, Linear SVM) showed similar feature importance distributions
3. Tree-based models provided more balanced positive/negative feature importance distributions
4. All models achieved at least 90% accuracy on the test set

## Visualization Outputs
- ROC curves
- Feature importance plots
- Model comparison plots
- Classification distribution plots
- Cross-validation plots (where applicable)

## Performance Metrics
Each model provides:
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC (where applicable)

## References
1. Breiman et al. (1984) - Original wave dataset
2. Rakotomalala (2005) - Dataset modification and initial analysis
3. Hsu et al. (2019) - Grafting technique implementation

## Contributing
This project welcomes contributions in:
- Implementation optimization
- Additional modeling approaches
- Enhanced visualization techniques
- Documentation improvements

## License
This project is available for academic and educational purposes.
