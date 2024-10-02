#### Importing Libraries

import pandas as pd  # Library for data manipulation and analysis, especially for importing and processing datasets
import matplotlib.pyplot as plt  # Library for creating visualizations
import seaborn as sns  # Data visualization library built on top of matplotlib, offering more complex visual patterns
import numpy as np  # Library for numerical operations, particularly useful for array manipulation and mathematical functions
from sklearn.model_selection import train_test_split  # Function to split the dataset into training and testing sets for model evaluation
from sklearn.neural_network import MLPRegressor  # Class for implementing a Multi-layer Perceptron Regressor for regression tasks
from sklearn.preprocessing import StandardScaler  # Tools for scaling data, StandardScaler for standardization and MinMaxScaler for normalization
from sklearn import metrics  # Library for accessing various metrics to evaluate machine learning models, such as MAE, MSE, and R2 Score




#### Loading the Dataset ####

# Path of the dataset that will be loaded into a dataframe
insurance = pd.read_csv('insurance.csv')




#### Data Inspection and Cleaning ####

# Display the first 5 rows to see how the dataframe is organizated
print(insurance.head())
# Print a concise summary of the dataframe
print(insurance.info())


# Check if there are any duplicate rows
duplicates = insurance[insurance.duplicated(keep=False)]
# Print the duplicate rows, if there are any
print(duplicates)
# Print the initial count of rows
print(f'Number of rows: {insurance.shape[0]}')
# Remove duplicate rows
insurance.drop_duplicates(inplace=True)
# Print the count of rows after removing duplicates
print(f'Number of rows: {insurance.shape[0]}')


# Display summary statistics for numerical columns in the dataframe
print(insurance.describe())

# Display summary statistics for categorical columns in the dataframe
print(insurance.describe(include='object'))
# Display the unique values for each categorical column in the dataframe
for _ in insurance.select_dtypes(include='object'):
   print(f"Unique values in '{_}':")
   print(insurance[_].unique())
   print()





################ Exploratory Data Analysis (EDA) ################


#### Visualizing a Correlation Matrix Between All Variables ####

# Note: pandas.corr() calculates the correlation between numeric columns, so we need to convert some categorical columns to numeric values to use this function effectively.

# Make a copy of the insurance dataframe
insurance_integer = insurance.copy()
# Convert 'sex', 'smoker', and 'region' to numeric codes
insurance_integer['sex'].replace({'female': 0, 'male': 1}, inplace=True)
insurance_integer['smoker'].replace({'no': 0, 'yes': 1}, inplace=True)
insurance_integer['region'].replace({'northwest': 0, 'northeast': 1, 'southwest': 2, 'southeast': 3}, inplace=True)

# Set up the correlation matrix
correlation_matrix = insurance_integer.corr()
# Print the correlation matrix
print(correlation_matrix)

# Generates a heatmap of the correlation between variables
ax = sns.heatmap(insurance_integer.corr(), annot=True, cmap=sns.cubehelix_palette(as_cmap=True))
ax.set_title('Correlation Between Variables')

# Show the figure
plt.show()
# Close the figure to free resources
plt.close()





#### Visualizing Univariate Categorical Data Distribution ####

# Count occurrences of each category
sex_counts = insurance['sex'].value_counts()
# Data to plot
labels = sex_counts.index
sizes = sex_counts.values

# Make the plot
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=['teal', 'orange'],
        autopct='%1.1f%%', startangle=140)

# Give the plot a title
plt.title('Proportion of Men and Women')
# Equal aspect ratio so that the pie is drawn as a circle
plt.axis('equal')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()



# Count occurrences of each category
smoker_counts = insurance['smoker'].value_counts()
# Data to plot
labels = smoker_counts.index
sizes = smoker_counts.values

# Make the plot
plt.figure(figsize=(6, 6))
plt.pie(sizes, labels=labels, colors=['springgreen', 'orchid'],
        autopct='%1.1f%%', startangle=140)

# Give the plot a title
plt.title('Proportion of Smokers and Non-Smokers')
# Equal aspect ratio so that the pie is drawn as a circle
plt.axis('equal')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()



# Count occurrences of each region
region_counts = insurance['region'].value_counts()

# Data to plot
labels = region_counts.index
sizes = region_counts.values

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(labels, sizes, color=['navy', 'orange', 'green', 'purple'])

# Add labels and title
plt.xlabel('Region')
plt.ylabel('Count')
plt.title('Distribution of Entries by Region')

# Calculate percentages
total_count = region_counts.sum()
percentages = (region_counts/total_count)*100

# Add percentage labels above each bar
for bar in bars:
    height = bar.get_height()
    percentage = (height/total_count)*100
    plt.text(bar.get_x() + bar.get_width()/2, height, f'{height} ({percentage:.1f}%)', 
             ha='center', va='bottom')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()





#### Visualizing Univariate Numerical Data Distribution ####

# Plot histogram for the 'age' column
plt.figure(figsize=(8, 5))
plt.hist(insurance['age'], bins=47, color='palegreen', edgecolor='black')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Distribution of Age')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()



# Plot histogram for the 'bmi' column
plt.figure(figsize=(8, 5))
plt.hist(insurance['bmi'], bins=75, color='palegreen', edgecolor='black')

# Add labels and title
plt.xlabel('BMI')
plt.ylabel('Frequency')
plt.title('Distribution of BMI')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()



# Plot histogram for the 'charges' column
plt.figure(figsize=(10, 6))
plt.hist(insurance['charges'], bins=100, color='palegreen', edgecolor='black')

# Add labels and title
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.title('Distribution of Insurance Charges')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()





#### Visualizing Univariate Numerical Data Across Categories ####

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the boxplot
ax.boxplot([insurance[insurance['region'] == region]['age'] for region in insurance['region'].unique()],
           positions=range(len(insurance['region'].unique())))

# Plot the violinplot
sns.violinplot(x='region', y='age', data=insurance, ax=ax, color='lightblue', inner=None)

# Add labels and title
ax.set_xticklabels(insurance['region'].unique())
ax.set_xlabel('Region')
ax.set_ylabel('Age')
ax.set_title('Age by Region')

# Show the plot
plt.show()
# Close the figure to free resources
plt.close()



# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define custom outlier properties for boxplot
flierprops = dict(marker='_', markersize=8)

# Plot the boxplot
ax.boxplot([insurance[insurance['region'] == region]['charges'] for region in insurance['region'].unique()],
           positions=range(len(insurance['region'].unique())), flierprops=flierprops)

# Plot the violinplot
sns.violinplot(x='region', y='charges', data=insurance, ax=ax, color='lightblue', inner=None)

# Set y-axis limit to start at 0
ax.set_ylim(0, insurance['charges'].max())

# Add labels and title
ax.set_xticklabels(insurance['region'].unique())
ax.set_xlabel('Region')
ax.set_ylabel('Charges')
ax.set_title('Charges by Region')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()



# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define custom outlier properties for boxplot
flierprops = dict(marker='_', markersize=8)

# Plot the boxplot
ax.boxplot([insurance[insurance['region'] == region]['bmi'] for region in insurance['region'].unique()],
           positions=range(len(insurance['region'].unique())), flierprops=flierprops)

# Plot the violinplot
sns.violinplot(x='region', y='bmi', data=insurance, ax=ax, color='lightblue', inner=None)

# Add labels and title
ax.set_xticklabels(insurance['region'].unique())
ax.set_xlabel('Region')
ax.set_ylabel('BMI')
ax.set_title('BMI by Region')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()



# Define age ranges (bins) and create a new column for age groups
bins = [18, 25, 35, 45, 55, 65]
labels = ['18-24', '25-34', '35-44', '45-54', '55-64']
insurance['age_range'] = pd.cut(insurance['age'], bins=bins, labels=labels, right=False)

# Create figure and axis
fig, ax = plt.subplots(figsize=(10, 6))

# Define custom outlier properties for the boxplot
flierprops = dict(marker='_', markersize=8)

# Plot the boxplot for BMI by age range
ax.boxplot([insurance[insurance['age_range'] == age]['bmi'] for age in labels],
           positions=range(len(labels)), flierprops=flierprops)

# Plot the violin plot for the same data
sns.violinplot(x='age_range', y='bmi', data=insurance, ax=ax, inner=None, color='lightblue', linewidth=1)

# Add labels and title
ax.set_xticklabels(labels)
ax.set_xlabel('Age range')
ax.set_ylabel('BMI')
ax.set_title('BMI by Age Range')

# Show plot
plt.show()
# Close the figure to free resources
plt.close()

# Drop the 'age_range' column
insurance.drop(columns=['age_range'], inplace=True)





#### Visualizing Bivariate Numerical Data ####

# Create a figure and three subplots (1 row, 3 columns)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 8))

# Hexbin plot on the first subplot
hb = ax1.hexbin(insurance['age'], insurance['children'], gridsize=30, cmap='cividis', mincnt=1)
ax1.set_xlabel('Age')
ax1.set_ylabel('Number of Children')
ax1.set_title('Hexbin plot')
fig.colorbar(hb, ax=ax1)

# Scatter plot with jitter on the second subplot
jitter_strength = 0.1  # Adjust this value to increase or decrease jitter
jittered_age = insurance['age'] + np.random.normal(0, jitter_strength, size=len(insurance))
jittered_children = insurance['children'] + np.random.normal(0, jitter_strength, size=len(insurance))
ax2.scatter(jittered_age, jittered_children, alpha=0.2)
ax2.set_xlabel('Age')
ax2.set_title('Scatter plot (with jitter)')

# KDE plot on the third subplot
sns.kdeplot(x='age', y='children', data=insurance, ax=ax3, cmap='cividis', fill=True)
ax3.set_xlabel('Age')
ax3.set_title('KDE plot')

# Set y-axis limits and ticks for the KDE plot
ax3.set_ylim(-1, 6)
ax3.set_yticks(range(6))

# Adjust the padding between and around subplots
plt.tight_layout()

# Show plot
plt.show()
# Close the figure to free resources
plt.close()



# Create a figure and three subplots (1 row, 3 columns)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 8))

# Hexbin plot on the first subplot
hb = ax1.hexbin(insurance['age'], insurance['bmi'], gridsize=30, cmap='cividis')
ax1.set_xlabel('Age')
ax1.set_ylabel('BMI')
ax1.set_title('Hexbin plot')
fig.colorbar(hb, ax=ax1)

# Scatter plot on the second subplot
ax2.scatter(insurance['age'], insurance['bmi'], alpha=0.3)
ax2.set_xlabel('Age')
ax2.set_title('Scatter plot')

# KDE plot on the third subplot
sns.kdeplot(data=insurance, x='age', y='bmi', ax=ax3, cmap='cividis', fill=True)
ax3.set_xlabel('Age')
ax3.set_title('KDE plot')

# Adjust the padding between and around subplots
plt.tight_layout()
# Show the plots
plt.show()
# Close the figure to free resources
plt.close()



# Create a figure and three subplots (1 row, 3 columns)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(27, 8))

# Hexbin plot on the first subplot
hb = ax1.hexbin(insurance['bmi'], insurance['charges'], gridsize=30, cmap='cividis')
ax1.set_xlabel('BMI')
ax1.set_ylabel('Charges')
ax1.set_title('Hexbin plot')
fig.colorbar(hb, ax=ax1)

# Scatter plot on the second subplot
ax2.scatter(insurance['bmi'], insurance['charges'], alpha=0.3)
ax2.set_xlabel('BMI')
ax2.set_title('Scatter plot')

# KDE plot on the third subplot
sns.kdeplot(data=insurance, x='bmi', y='charges', ax=ax3, cmap='cividis', fill=True)
ax3.set_xlabel('BMI')
ax3.set_title('KDE plot')

# Adjust the padding between and around subplots
plt.tight_layout()
# Show the plots
plt.show()
# Close the figure to free resources
plt.close()





################ Comprehensive Analysis of Factors Affecting Cost ################


# Sort the dataframe first by 'age', then by 'smoker', and finally by 'bmi'
sorted_insurance_age_smoker_bmi = insurance.sort_values(by=['age', 'smoker', 'bmi'])
# Display the sorted dataframe
print(sorted_insurance_age_smoker_bmi.head(25))


# Sort the dataframe by the 'charges' column
sorted_insurance_charges = insurance.sort_values(by=['charges'])
# Display the sorted datafrane
print(sorted_insurance_charges.head(25))





#### Visualizing Multivariate Numerical Data ####

# Create a scatter plot with seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(data=insurance, x='age', y='charges', hue='sex', palette={'male': 'blue', 'female': 'orange'}, alpha=0.4)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs Charges by Gender')

# Show the plot
plt.show()
# Close the figure to free resources
plt.close()



# Create a figure for the scatter plot
plt.figure(figsize=(15, 8))

# Scatter plot with age represented by color
scatter = plt.scatter(insurance['age'], insurance['charges'], c=insurance['bmi'], cmap='viridis', alpha=0.4, vmin=20, vmax=40) # Limiting the BMI between 20 and 45, as there are few people outside this range
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs Charges (with BMI as color)')
plt.colorbar(scatter, label='BMI')

# Show the plot
plt.show()
# Close the figure to free resources
plt.close()



# Create a scatter plot with seaborn
plt.figure(figsize=(12, 8))
sns.scatterplot(data=insurance, x='age', y='charges', hue='smoker', palette={'yes': 'fuchsia', 'no': 'springgreen'}, alpha=0.4)

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs Charges by Smoker Status')

# Show the plot
plt.show()
# Close the figure to free resources
plt.close()



# Create a figure for the scatter plot
plt.figure(figsize=(15, 8))

# Scatter plot with age represented by color
scatter = plt.scatter(insurance['bmi'], insurance['charges'], c=insurance['age'], cmap='viridis', alpha=0.4)
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI vs Charges (with Age as color)')
plt.colorbar(scatter, label='Age')

# Show the plot
plt.show()
# Close the figure to free resources
plt.close()



# Make a copy of the insurance dataframe
insurance_nonsmokers = insurance.copy()
# Filter so the dataframe only has non-smokers
insurance_nonsmokers = insurance_nonsmokers[insurance_nonsmokers['smoker'] == 'no']

# Create a figure for the scatter plot
plt.figure(figsize=(15, 8))

# Scatter plot with age represented by color
scatter = plt.scatter(insurance_nonsmokers['bmi'], insurance_nonsmokers['charges'], c=insurance_nonsmokers['age'], cmap='viridis', alpha=0.4)
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI vs Charges (Non-Smokers) (with Age as color)')
plt.colorbar(scatter, label='Age')

# Show the plot
plt.show()
# Close the figure to free resources
plt.close()



# Make a copy of the insurance dataframe
insurance_smokers = insurance.copy()
# Filter so the dataframe only has smokers
insurance_smokers = insurance_smokers[insurance_smokers['smoker'] == 'yes']

# Create a figure for the scatter plot
plt.figure(figsize=(15, 8))

# Scatter plot with age represented by color
scatter = plt.scatter(insurance_smokers['bmi'], insurance_smokers['charges'], c=insurance_smokers['age'], cmap='viridis', alpha=0.4)
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI vs Charges (Smokers) (with Age as color)')
plt.colorbar(scatter, label='Age')

# Show the plot
plt.show()
# Close the figure to free resources
plt.close()



# Create two new DataFrames for BMI filtering
insurance_smokers_bmi_under_30 = insurance_smokers[insurance_smokers['bmi'] <= 30]
insurance_smokers_bmi_over_30 = insurance_smokers[insurance_smokers['bmi'] > 30]

# Create a figure for the scatter plots
plt.figure(figsize=(15, 6))

# Scatter plot for smokers with BMI 30 or under
plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
scatter1 = plt.scatter(insurance_smokers_bmi_under_30['bmi'],
                        insurance_smokers_bmi_under_30['charges'],
                        c=insurance_smokers_bmi_under_30['age'], 
                        cmap='viridis', alpha=0.4)
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI vs Charges (Smokers): BMI <= 30')
plt.colorbar(scatter1, label='Age')

# Scatter plot for smokers with BMI over 30
plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
scatter2 = plt.scatter(insurance_smokers_bmi_over_30['bmi'],
                        insurance_smokers_bmi_over_30['charges'],
                        c=insurance_smokers_bmi_over_30['age'], 
                        cmap='viridis', alpha=0.4)
plt.xlabel('BMI')
plt.ylabel('Charges')
plt.title('BMI vs Charges (Smokers): BMI > 30')
plt.colorbar(scatter2, label='Age')

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()
# Close the figure to free resources
plt.close()





################ Dealing with Outliers ################

# Defining a function that filters outliers
def remove_outliers(dataframe, y):
    # Calculate the average cost for each age
    avg_cost_age = dataframe.groupby('age')['charges'].mean()
    # Create a mask for the entire DataFrame
    mask = pd.Series(True, index=dataframe.index)  # Start with all True

    # Iterate over each age
    for age, avg_cost in avg_cost_age.items():
        # Identify outliers based on the threshold
        outliers = dataframe[(dataframe['age'] == age) & (dataframe['charges'] > y * avg_cost)].index
        # Update the mask to exclude outliers
        mask[outliers] = False

    # Return a new DataFrame without outliers
    return dataframe[mask]



# Test some different 'y' threshold values
y = [1.5, 1.4, 1.39, 1.35]

# Create a figure for the subplots
plt.figure(figsize=(15, 12))

# Loop through the thresholds and create subplots
for index, threshold in enumerate(y):
    # Make a copy of the original dataframe for each threshold
    temp_insurance_nonsmokers = insurance_nonsmokers.copy()

    # Apply the function to remove outliers
    temp_insurance_nonsmokers = remove_outliers(temp_insurance_nonsmokers, threshold)

    # Create a subplot in the 2x2 grid
    plt.subplot(2, 2, index + 1)  # 2 rows, 2 columns, subplot index starts at 1
    sns.scatterplot(data=temp_insurance_nonsmokers, x='age', y='charges', hue='bmi', palette='viridis', alpha=0.4)

    # Add labels and title
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title(f'Age vs Charges (Non-Smokers): y = {threshold}')

    # Move the legend to the top left corner
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()
# Close the figure to free resources
plt.close()



# Test some different 'y' threshold values
y = [1.3, 1.25, 1.24, 1.2]

# Create a figure for the subplots
plt.figure(figsize=(15, 12))

# Loop through the thresholds and create subplots
for index, threshold in enumerate(y):
    # Make a copy of the original dataframe for each threshold
    temp_insurance_smokers_bmi_under_30 = insurance_smokers_bmi_under_30.copy()

    # Apply the function to remove outliers
    temp_insurance_smokers_bmi_under_30 = remove_outliers(temp_insurance_smokers_bmi_under_30, threshold)

    # Create a subplot in the 2x2 grid
    plt.subplot(2, 2, index + 1)  # 2 rows, 2 columns, subplot index starts at 1
    sns.scatterplot(data=temp_insurance_smokers_bmi_under_30, x='age', y='charges', hue='bmi', palette='viridis', alpha=0.4)

    # Add labels and title
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title(f'Age vs Charges (Smokers with BMI <= 30): y = {threshold}')

    # Move the legend to the top left corner
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()
# Close the figure to free resources
plt.close()



# Test some different 'y' threshold values
y = [1.25, 1.11, 1.09, 1.05]

# Create a figure for the subplots
plt.figure(figsize=(15, 12))

# Loop through the thresholds and create subplots
for index, threshold in enumerate(y):
    # Make a copy of the original dataframe for each threshold
    temp_insurance_smokers_bmi_over_30 = insurance_smokers_bmi_over_30.copy()

    # Apply the function to remove outliers
    temp_insurance_smokers_bmi_over_30 = remove_outliers(temp_insurance_smokers_bmi_over_30, threshold)

    # Create a subplot in the 2x2 grid
    plt.subplot(2, 2, index + 1)  # 2 rows, 2 columns, subplot index starts at 1
    sns.scatterplot(data=temp_insurance_smokers_bmi_over_30, x='age', y='charges', hue='bmi', palette='viridis', alpha=0.4)

    # Add labels and title
    plt.xlabel('Age')
    plt.ylabel('Charges')
    plt.title(f'Age vs Charges (Smokers with BMI > 30): y = {threshold}')

    # Move the legend to the top left corner
    plt.legend(loc='upper left', bbox_to_anchor=(0, 1))

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()
# Close the figure to free resources
plt.close()





################ Multi-layer Perceptron Regressor (MLPRegressor) ################


#### Dataset Preparation ####

# Filter the outliers
insurance_nonsmokers = remove_outliers(insurance_nonsmokers, 1.39)
insurance_smokers_bmi_under_30 = remove_outliers(insurance_smokers_bmi_under_30, 1.24)
insurance_smokers_bmi_over_30 = remove_outliers(insurance_smokers_bmi_over_30, 1.09)

# Create the insurance dataframe with outliers filtered out
insurance_filtered = pd.concat([insurance_nonsmokers, insurance_smokers_bmi_under_30, insurance_smokers_bmi_over_30], ignore_index=True)

# Make a copy of the insurance dataframe
insurance_filtered_integer = insurance_filtered.copy()

# Convert 'sex', 'smoker', and 'region' to numeric codes
insurance_filtered_integer['sex'].replace({'female': 0, 'male': 1}, inplace=True)
insurance_filtered_integer['smoker'].replace({'no': 0, 'yes': 1}, inplace=True)
insurance_filtered_integer['region'].replace({'northwest': 0, 'northeast': 1, 'southwest': 2, 'southeast': 3}, inplace=True)

# Note: 'insurance_integer' is the original dataframe, 
# 'insurance_filtered_integer' is the dataframe without outliers, 
# both with categorical columns' values converted to numeric codes.

# Print the size of each dataframe
print(f'insurance_integer: {insurance_integer.shape}')
print(f'insurance_filtered_integer: {insurance_filtered_integer.shape}')




#### Feature and Target Variable Separation ####

# Exclusion of the 'charges' variable from the input features
x_original = insurance_integer.drop('charges', axis=1)
x_filtered = insurance_filtered_integer.drop('charges', axis=1)
# Inclusion of the 'charges' variable in the desired output vector
y_original = insurance_integer['charges']
y_filtered = insurance_filtered_integer['charges']
# Normalization of the 'charges' variable to ensure all variables are on the same scale
maxY_original = y_original.max()
maxY_filtered = y_filtered.max()
y_original = y_original / maxY_original
y_filtered = y_filtered / maxY_filtered

# Data standardization
scaler1 = StandardScaler().fit(x_original)  # Creates the first model for fitting
scaler2 = StandardScaler().fit(x_filtered)  # Creates the second model for fitting

# Applying standardization
x_original = scaler1.transform(x_original)  # Applies standardization to the 'x' of the original dataset
x_filtered = scaler2.transform(x_filtered)  # Applies standardization to the 'x' of the filtered dataset

# Splitting the dataset into training and testing samples, allocating 30% of the data for testing
o_trainX, o_testX, o_trainY, o_testY = train_test_split(x_original, y_original, test_size=0.3)
f_trainX, f_testX, f_trainY, f_testY = train_test_split(x_filtered, y_filtered, test_size=0.3)




#### Model Initialization and Training ####

# Creation of the first neural network model (MLPRegressor) for the original dataset
mlp_reg_original = MLPRegressor(hidden_layer_sizes=(3), activation='logistic', solver='adam',
                       max_iter=10000, tol=0.0000001, momentum=0.5, early_stopping=True, epsilon=1e-08,
                       n_iter_no_change=100, random_state=0)

# Creation of the second neural network model (MLPRegressor) for the filtered dataset (without outliers)
mlp_reg_filtered = MLPRegressor(hidden_layer_sizes=(3), activation='logistic', solver='adam',
                       max_iter=10000, tol=0.0000001, momentum=0.5, early_stopping=True, epsilon=1e-08,
                       n_iter_no_change=100, random_state=0)

# Training the neural networks using the training data (input and output)
mlp_reg_original.fit(o_trainX, o_trainY)  # Fit the original model
mlp_reg_filtered.fit(f_trainX, f_trainY)  # Fit the filtered model




#### Visualizing Model Predictions ####

# Prediction for the original dataset
o_predY = mlp_reg_original.predict(o_testX)  # Predicting charges using the original model
df_temp = pd.DataFrame({'Desired': o_testY, 'Estimated': o_predY})  # Create DataFrame for comparison
df_temp = df_temp.head(40)  # Select the first 40 rows for plotting
df_temp.plot(kind='bar', figsize=(10, 6))  # Plotting predicted vs actual charges
plt.title("Predicted vs Actual Charges (Original Dataset)", fontsize=16)  # Title for the original dataset
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')  # Major grid lines
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')   # Minor grid lines
plt.show()  # Show the plot
plt.close()  # Close the figure to free resources

# Prediction for the filtered dataset
f_predY = mlp_reg_filtered.predict(f_testX)  # Predicting charges using the filtered model
df_temp = pd.DataFrame({'Desired': f_testY, 'Estimated': f_predY})  # Create DataFrame for comparison
df_temp = df_temp.head(40)  # Select the first 40 rows for plotting
df_temp.plot(kind='bar', figsize=(10, 6))  # Plotting predicted vs actual charges
plt.title("Predicted vs Actual Charges (Filtered Dataset)", fontsize=16)  # Title for the filtered dataset
plt.grid(which='major', linestyle='-', linewidth='0.5', color='gray')  # Major grid lines
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='blue')   # Minor grid lines
plt.show()  # Show the plot
plt.close()  # Close the figure to free resources




#### Training Loss Evaluation ####

# Creating subplots with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Evaluating the training error for the original model
axes[0].plot(mlp_reg_original.loss_curve_)  # Plot of loss values during the training process
axes[0].set_title("Loss Curve during Training (Original Dataset)", fontsize=14)  # Title for the original model's plot
axes[0].set_xlabel('Epochs')  # Label for the x-axis
axes[0].set_ylabel('Cost')    # Label for the y-axis

# Evaluating the training error for the filtered model
axes[1].plot(mlp_reg_filtered.loss_curve_)  # Plot of loss values during the training process
axes[1].set_title("Loss Curve during Training (Filtered Dataset)", fontsize=14)  # Title for the filtered model's plot
axes[1].set_xlabel('Epochs')  # Label for the x-axis
axes[1].set_ylabel('Cost')    # Label for the y-axis

# Adjust layout to prevent overlap
plt.tight_layout()
# Show the plots
plt.show()
# Close the figure to free resources
plt.close()




#### Performance Metrics (Original Dataset) ####

# Metrics for the neural network MLPRegressor trained with the original dataset
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(o_testY, o_predY))  # Ranges from 0 to infinity; lower values indicate better performance.
print('Mean Squared Error (MSE):', metrics.mean_squared_error(o_testY, o_predY))  # Ranges from 0 to infinity; lower values indicate better performance.
print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(o_testY, o_predY, squared=False))  # Ranges from 0 to infinity; lower values indicate better performance.
print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(o_testY, o_predY))  # Represents the percentage error relative to the actual values.
print('R2: ', metrics.r2_score(o_testY, o_predY))  # Represents the R2 Score; ranges from 0 to 1. An R2 of 1 indicates a perfect linear relationship with the data.




#### Performance Metrics (Filtered Dataset) ####

# Metrics for the neural network MLPRegressor trained with the filtered dataset
print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(f_testY, f_predY))  # Ranges from 0 to infinity; lower values indicate better performance.
print('Mean Squared Error (MSE):', metrics.mean_squared_error(f_testY, f_predY))  # Ranges from 0 to infinity; lower values indicate better performance.
print('Root Mean Squared Error (RMSE):', metrics.mean_squared_error(f_testY, f_predY, squared=False))  # Ranges from 0 to infinity; lower values indicate better performance.
print('Mean Absolute Percentage Error (MAPE):', metrics.mean_absolute_percentage_error(f_testY, f_predY))  # Represents the percentage error relative to the actual values.
print('R2: ', metrics.r2_score(f_testY, f_predY))  # Represents the R2 Score; ranges from 0 to 1. An R2 of 1 indicates a perfect linear relationship with the data.





################ Conclusion ################

'''
The project successfully demonstrated a comprehensive approach to data quality and analysis. Through meticulous data inspection, manipulation, and cleaning, we ensured the
integrity of the dataset. The exploratory data analysis (EDA) revealed valuable insights and patterns, guiding our understanding of the data's characteristics.

We effectively utilized a variety of visualization techniques, including bar charts, heatmaps, pie charts, histograms, boxplots, violin plots, scatter plots, hexbin plots,
and KDE, to analyze both categorical and numerical variables across univariate, bivariate, and multivariate representations.

Additionally, the creation and application of an outlier filtering function, based on EDA conclusions and age-specific cost averages, allowed us to remove extreme values while
preserving the data's main trends. This preprocessing step significantly enhanced the performance of our predictive model.

The implementation of the Multi-layer Perceptron Regressor (MLPRegressor) showcased the model's capability in forecasting based on different datasets. Notably, the application
of outlier filtering improved the model's accuracy from approximately 84% to an impressive 99%, emphasizing the importance of data preprocessing in machine learning tasks.

Overall, this project highlights the critical role of data quality and exploratory analysis in building robust predictive models, paving the way for more accurate decision-making
based on reliable information.
'''