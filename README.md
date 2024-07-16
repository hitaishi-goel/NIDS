# Supervised Machine Learning Classification for Network Traffic Analysis

## Overview

This project focuses on building and evaluating various machine learning models for network intrusion detection using the NSL-KDD dataset. The dataset provides network traffic data and is commonly used for benchmarking intrusion detection systems. The goal of this project is to classify network traffic as either "normal" or "attack" using a range of classification algorithms and to compare their performance based on accuracy, precision, and recall.

## Dataset

- **Dataset**: [NSL-KDD](https://www.unb.ca/cic/datasets/nsl-kdd.html)
- **Description**: The NSL-KDD dataset is an enhanced version of the KDD Cup 1999 dataset for network intrusion detection. It includes a set of features that describe network traffic and attack types.
- **Data Source**: `KDDTrain.txt` (Training dataset)

## Project Structure

```plaintext
.
├── README.md
├── KDDTrain.txt
├── notebook.ipynb
└── requirements.txt
```

- `README.md`: This file.
- `KDDTrain.txt`: The NSL-KDD dataset training data.
- `notebook.ipynb`: Jupyter notebook containing the code for data preprocessing, model training, evaluation, and comparison.
- `requirements.txt`: List of required packages for the project.

## Requirements

To set up the project environment, create a virtual environment and install the required packages:

```bash
pip install -r requirements.txt
```

**`requirements.txt`**:

```plaintext
numpy
pandas
scikit-learn
matplotlib
seaborn
tensorflow
xgboost
lightgbm
catboost
```

## Code Explanation

### 1. **Data Preprocessing**

   - **Load Dataset**: Load the training data from `KDDTrain.txt`.
   - **Feature Selection**: Drop irrelevant features and scale numerical features.
   - **Label Encoding**: Convert attack types into binary classification: `normal` (0) and `attack` (1).
   - **Feature Encoding**: Apply one-hot encoding to categorical features.

### 2. **Model Training and Evaluation**

   The following models are trained and evaluated:

   - **Random Forest Classifier**: Ensemble method using multiple decision trees.
   - **Logistic Regression**: Linear model for binary classification.
   - **Decision Tree Classifier**: A model that splits data based on feature values.
   - **K Nearest Neighbors (KNN)**: Classifies based on the majority vote from the nearest neighbors.
   - **Support Vector Classifier (LinearSVC)**: Finds the hyperplane that best separates classes.
   - **Extreme Gradient Boosting (XGBoost)**: Boosted tree model for classification.
   - **Light Gradient Boosting Machine (LGBM)**: Gradient boosting with a focus on efficiency.
   - **CatBoost Classifier**: Gradient boosting model designed for categorical features.
   - **AdaBoost Classifier**: Ensemble method that combines weak classifiers.
   - **Ridge Classifier**: Linear model with L2 regularization.

   **Evaluation Metrics**:
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix

### 3. **Results**

   The results are visualized using a line plot comparing the training and test accuracies of the models.

### 4. **Visualization**

   - **Accuracy Comparison Plot**: Shows the performance of each model in terms of training and test accuracy.

## How to Run

1. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare the Dataset**:
   - Download the NSL-KDD dataset and place `KDDTrain.txt` in the project directory.

3. **Execute the Notebook**:
   - Open `notebook.ipynb` in Jupyter Notebook or JupyterLab and run the cells to perform data preprocessing, model training, and evaluation.

## Results

The results of the different models are compared based on accuracy, precision, and recall. The performance metrics are summarized in a plot.

## References

- [NSL-KDD Dataset](https://www.unb.ca/cic/datasets/nsl-kdd.html)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/en/latest/)
- [CatBoost Documentation](https://catboost.ai/docs/)

## Contributing

Feel free to fork the repository and submit pull requests with improvements or bug fixes.

Feel free to customize this `README.md` based on your specific requirements or any additional details about the project!
