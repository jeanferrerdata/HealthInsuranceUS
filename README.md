# US Medical Insurance Data Analysis

<p style="font-size: 1.2em; line-height: 1.3;">
In this project, we analyze a medical insurance dataset to understand how various factors influence insurance costs.<br>
Additionally, we aim to employ the Multi-layer Perceptron Regressor (MLPRegressor), a type of neural network used in machine learning, to predict the insurance cost for individuals based on these factors.
</p>

## Table of Contents

- [Installation](#installation)
- [Loading the Dataset](#loading-the-dataset)
- [Data Inspection and Cleaning](#data-inspection-and-cleaning)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
  - [Visualizing a Correlation Matrix Between All Variables](#visualizing-a-correlation-matrix-between-all-variables)
  - [Visualizing Univariate Categorical Data Distribution](#visualizing-univariate-categorical-data-distribution)
  - [Visualizing Univariate Numerical Data Distribution](#visualizing-univariate-numerical-data-distribution)
  - [Visualizing Univariate Numerical Data Across Categories](#visualizing-univariate-numerical-data-across-categories)
  - [Visualizing Bivariate Numerical Data](#visualizing-bivariate-numerical-data)
- [Comprehensive Analysis of Factors Affecting Cost](#comprehensive-analysis-of-factors-affecting-cost)
  - [Visualizing Multivariate Numerical Data](#visualizing-multivariate-numerical-data)
  - [Dealing with Outliers](#dealing-with-outliers)
- [Multi-layer Perceptron Regressor (MLPRegressor)](#multi-layer-perceptron-regressor-mlpregressor)
  - [Dataset Preparation](#dataset-preparation)
  - [Feature and Target Variable Separation](#feature-and-target-variable-separation)
  - [Model Initialization and Training](#model-initialization-and-training)
  - [Visualizing Model Predictions](#visualizing-model-predictions)
  - [Training Loss Evaluation](#training-loss-evaluation)
  - [Performance Metrics (Original Dataset)](#performance-metrics-original-dataset)
  - [Performance Metrics (Filtered Dataset)](#performance-metrics-filtered-dataset)
- [Conclusion](#conclusion)

---

## Installation

To run this project, you'll need to install the necessary libraries. You can do this by pip installing the following:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

Otherwise, you could use the Anaconda environment.

## Loading the Dataset
The dataset used is the US Health Insurance dataset. Ensure you have the dataset file named insurance.csv in your working directory.  
It is loaded into a pandas dataframe, containing medical insurance records with columns for age, sex, BMI, children, smoker status, region, and charges.

## Data Inspection and Cleaning
This step involves checking for missing or inconsistent data and cleaning it up to prepare for further analysis.  
Some tasks include:
- Displaying the first rows.
- Printing a concise summary of the dataframe.
- Identifying null values.
- Checking and removing duplicate rows.

## Exploratory Data Analysis (EDA)
This phase aims to uncover insights and patterns from the data.

### Visualizing a Correlation Matrix Between All Variables
A correlation matrix is visualized to identify relationships between variables.  
This matrix helps to quickly spot highly correlated variables that might affect insurance costs.

### Visualizing Univariate Categorical Data Distribution
Bar and pie charts are used to show the distribution of categorical variables, such as `sex`, `smoker`, and `region`.

### Visualizing Univariate Numerical Data Distribution
Histograms are used to analyze the distribution patterns of numerical variables like `age`, `bmi`, and `charges`.

### Visualizing Univariate Numerical Data Across Categories
This section explores how numerical variables (e.g., `age`, `bmi`, and `charges`) vary across categories like `region`.  
Boxplots and violin plots are used for visualization.

### Visualizing Bivariate Numerical Data
Scatter, hexbin and KDE plots are employed to explore relationships between two numerical variables.  

## Comprehensive Analysis of Factors Affecting Cost
This section focuses on more complex interactions and trends that affect insurance costs.

### Visualizing Multivariate Numerical Data
Scatter plots with color hues are used to illustrate how multiple numerical variables interact and influence insurance costs.

### Dealing with Outliers
Outliers in the dataset, specifically in the charges variable, were addressed using a custom function.  
The `remove_outliers` function filters out values that significantly exceed the average insurance cost for each individual age.

The function works as follows:
- It calculates the average insurance charges for each age group.
- For each age, any data point where the charges exceed a certain threshold (defined by the parameter `y` times the average) is marked as an outlier.
- The final output is a dataframe with outliers removed, ensuring that the model's predictions are not skewed by extreme values.

## Multi-layer Perceptron Regressor (MLPRegressor)
This phase involves building a neural network model to predict insurance costs.

### Dataset Preparation
The dataset is filtered to remove outliers from the `charges` variable.  
Categorical variables (`sex`, `smoker`, `region`) are converted to numeric codes.

### Feature and Target Variable Separation
The dataset is split into features (`age`, `sex`, `bmi`, `children`, `smoker`, `region`) and the target variable (`charges`).  
The `charges` variable is normalized for proper scaling, and the features are standardized to ensure they have a mean of 0 and a standard deviation of 1.  
Finally, the dataset is split into training and testing samples, with 30% allocated for testing.

### Model Initialization and Training
The MLPRegressor model is initialized with selected hyperparameters and trained on both the original and filtered dataframes to predict insurance costs.

### Visualizing Model Predictions
After training, the model's predictions are visualized against actual values to assess performance.  
A double bar graph provides a quick view of how well the model is performing.

### Training Loss Evaluation
During the training process, the loss curve is visualized to monitor how well the model is learning and adjusting.  
This curve shows whether the model converges properly.

### Performance Metrics (Original Dataset)
To evaluate the model's performance, several metrics are calculated:
- **Mean Absolute Error (MAE)**
- **Mean Squared Error (MSE)**
- **Root Mean Squared Error (RMSE)**
- **Mean Absolute Percentage Error (MAPE)**
- **R² Score**

These metrics give a clear picture of how accurate the predictions are on the original dataset.

### Performance Metrics (Filtered Dataset)
The same performance evaluation is conducted on the filtered dataset (with outliers removed) to compare and understand the impact of outlier removal on model accuracy.

## Conclusion
This project demonstrated a comprehensive approach to data quality and analysis, ensuring the integrity of the dataset through meticulous inspection, manipulation, and cleaning. The exploratory data analysis (EDA) revealed valuable insights, utilizing various visualization techniques to analyze categorical and numerical variables.  

The development of an outlier filtering function, based on EDA conclusions and age-specific cost averages, effectively removed extreme values while preserving key trends, significantly enhancing our predictive model's performance.  

The implementation of the Multi-layer Perceptron Regressor (MLPRegressor) showcased the model's capability in forecasting based on different datasets. Notably, the application of outlier filtering improved the model's accuracy from approximately 84% to an impressive 99%, emphasizing the importance of data preprocessing in machine learning tasks.  

Overall, this project highlights the critical role of data quality and exploratory analysis in building robust predictive models, paving the way for more accurate decision-making based on reliable information.

--- 

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
