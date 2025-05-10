# Diabetes Prediction with Machine Learning Models

This repository implements a suite of machine learning models to predict diabetes using the [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database). The project compares various algorithms, including Decision Trees, Random Forests, Gradient Boosting, Multilayer Perceptrons (MLP), XGBoost, Ensemble methods, Support Vector Machines (SVM), and more to come. Each model is implemented with data preprocessing, hyperparameter tuning, and comprehensive evaluation metrics.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Models Implemented](#models-implemented)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
The goal of this project is to build and compare machine learning models for predicting diabetes based on clinical features such as `Glucose`, `BMI`, `Age`, and others. The repository includes scripts for preprocessing the dataset (handling missing/invalid data, scaling features, balancing classes with SMOTE), training models, tuning hyperparameters, evaluating performance (accuracy, precision, recall, F1-score, ROC-AUC), and visualizing results (e.g., Decision Tree plots, feature importance).

Key features:
- Robust data preprocessing to handle zero values and class imbalance.
- Hyperparameter optimization using GridSearchCV.
- Comprehensive evaluation with classification reports and confusion matrices.
- Modular code structure for easy extension to new models.
- Visualizations for model interpretability.

## Dataset
The Pima Indians Diabetes Dataset contains 768 samples with 8 features and a binary target (`Outcome`):
- **Features**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- **Target**: `Outcome` (0 = non-diabetic, 1 = diabetic)
- **Source**: Available in the `data/` directory as `diabetes.csv`.

**Note**: The `Insulin` feature is dropped due to frequent missing/invalid values, and rows with zero values for `Glucose`, `BloodPressure`, or `BMI` are filtered out.

## Repository Structure
```
diabetes-ml-models/
├── data/
│   └── diabetes.csv              # Pima Indians Diabetes Dataset
├── 0_decision_tree.py           # Decision Tree Classifier
├── 1_random_forest.py           # Random Forest Classifier
├── 2_gradient_boost.py          # Gradient Boosting Classifier
├── 3_mlp.py                     # Multilayer Perceptron (Neural Network)
├── 4_xg_boost.py                # XGBoost Classifier
├── 5_ensemble.py                # Ensemble of multiple models
├── 6_svm.py                     # Support Vector Machine Classifier
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

- Each `.py` file contains a complete pipeline: data loading, preprocessing, model training, evaluation, and visualization (where applicable).
- Additional model scripts will be added under `7_*.py`, `8_*.py`, etc.

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/diabetes-ml-models.git
   cd diabetes-ml-models
   ```

2. **Set Up a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Dataset**:
   - Ensure `diabetes.csv` is in the `data/` directory. If not, download it from [Kaggle](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) and place it in `data/`.

## Usage
1. **Run a Specific Model**:
   - Execute any model script using Python:
     ```bash
     python 0_decision_tree.py
     ```
   - Each script will:
     - Load and preprocess `diabetes.csv`.
     - Train the model with hyperparameter tuning (if applicable).
     - Print evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix).
     - Display visualizations (e.g., Decision Tree plot, feature importance).

2. **Example Output** (for `0_decision_tree.py`):
   ```
   Best Parameters: {'max_depth': 5, 'min_samples_leaf': 2, 'min_samples_split': 5}
   Feature Importance:
       Feature       Importance
       Glucose       0.45
       BMI           0.25
       Age           0.15
       ...
   SIMPLE DECISION TREE
   ===========================================================================
   Classification Report:
                  precision    recall  f1-score   support
   non-diabetic    0.78      0.82      0.80        80
   diabetic        0.68      0.62      0.65        50
   ...
   ```

3. **Compare Models**:
   - Run all scripts and compare their evaluation metrics to determine the best-performing model.
   - The `5_ensemble.py` script combines multiple models for potentially improved performance.

## Models Implemented
The repository currently includes the following models:
1. **Decision Tree** (`0_decision_tree.py`):
   - Simple tree-based model with GridSearchCV for hyperparameter tuning.
   - Visualizes the tree structure.
2. **Random Forest** (`1_random_forest.py`):
   - Ensemble of Decision Trees for improved robustness.
   - Reports feature importance.
3. **Gradient Boosting** (`2_gradient_boost.py`):
   - Boosting algorithm for sequential tree improvement.
4. **Multilayer Perceptron (MLP)** (`3_mlp.py`):
   - Neural network with tunable layers and neurons.
5. **XGBoost** (`4_xg_boost.py`):
   - Optimized gradient boosting for high performance.
6. **Ensemble** (`5_ensemble.py`):
   - Combines predictions from multiple models (e.g., Voting Classifier).
7. **Support Vector Machine (SVM)** (`6_svm.py`):
   - Kernel-based classifier for non-linear boundaries.

**More to Come**:
- Additional models (e.g., Logistic Regression, KNN, LightGBM) will be added as `7_*.py`, `8_*.py`, etc.
- Scripts for model comparison and visualization of performance metrics.

## Future Work
- Add more models (e.g., Logistic Regression, K-Nearest Neighbors, LightGBM).
- Implement cross-model comparison with a unified evaluation script.
- Visualize performance metrics (e.g., ROC curves, precision-recall curves) across models.
- Explore feature engineering (e.g., polynomial features, interaction terms).
- Add unit tests for preprocessing and model training functions.
- Integrate CI/CD with GitHub Actions for automated testing and dependency updates.

## Contributing
Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/new-model`).
3. Add your model script (e.g., `7_logistic_regression.py`) following the existing structure.
4. Update `README.md` and `requirements.txt` if new dependencies are added.
5. Submit a pull request with a clear description of your changes.

Please ensure code follows PEP 8 style guidelines and includes comments for clarity.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**Author**: Shounak Bhalerao  
**Contact**: shounakbhalerao@gmail.com  
**Last Updated**: May 9, 2025