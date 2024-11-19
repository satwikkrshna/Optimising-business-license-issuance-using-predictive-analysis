#%%
import warnings 
warnings.simplefilter(action='ignore' , category=FutureWarning)

#%%
#Data Preprocessing 
import pandas as pd 
file_path = r'C:\Users\pvsat\OneDrive\Desktop\KOVENDAN\ML PROJECT\New folder'

data = pd.read_csv(file_path)

# to know the name of the attributes 
print ('-------------------------------------------------')
print ('Attribute names of the dataframe')
print ('--------------------------------------------------')
print (data.columns)
print ('--------------------------------------------------')

# head of the data frame 
print ('----------------------------------------------------')
print ('top 5 obs of the data')
print ('----------------------------------------------------')
print (data.head())
print ('-------------------------------------------------')

# Tail of the data frame 

print ('---------------------------------------------------')
print ('last 5 obs of the data')
print ('---------------------------------------------------')
print (data.tail())
print ('---------------------------------------------------')
data.info()

#%%
#creating duplicate dataset 
df = data.copy()

#%%
#droping the unwanted column 
df = df.drop(columns=['WEEK', 'LICENCE CODE'])

#%%
# to know the license descirption 
# List Original Categories
original_categories = data['LICENSE DESCRIPTION'].unique()

print("Original Categories in 'LICENSE DESCRIPTION':")
for category in original_categories:
    print(category)

#%%
# missing values conerting WEEK variable 

missing_value = df.isnull(). sum()
ms_percentage = (df.isnull(). sum()/(len(df)))*100
Missing_datainfo = pd.DataFrame({'total missings':missing_value,'percentage' : ms_percentage})

print ('----------------------------------------------------')
print ('Updates data information')
print ('----------------------------------------------------')
print (Missing_datainfo)
print('-------------------------------------------------------')

#imputing the missing values 

from sklearn.impute import SimpleImputer 

df['AVERAGE DAYS TO ISSUE LICENSE'] = pd.to_numeric(df['AVERAGE DAYS TO ISSUE LICENSE'], errors='coerce')
df['MEDIAN DAYS TO ISSUE LICENSE'] = pd.to_numeric(df['MEDIAN DAYS TO ISSUE LICENSE'], errors='coerce')
df['TOTAL LICENSES ISSUED'] = pd.to_numeric(df['TOTAL LICENSES ISSUED'], errors='coerce')


numerical_cols = ['AVERAGE DAYS TO ISSUE LICENSE', 'MEDIAN DAYS TO ISSUE LICENSE', 'TOTAL LICENSES ISSUED']
categorical_cols = ['LICENSE DESCRIPTION']

#imputing catergorical variable 'mode'
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_cols] = imputer_cat.fit_transform(df[categorical_cols])

#imputing numerical variable 'mean'
imputer_num = SimpleImputer(strategy='mean')
df[numerical_cols] = imputer_num.fit_transform(df[numerical_cols])


# second set of prepared MS values
missing_value_m2 = df.isnull(). sum() 
ms_percentage = (df.isnull(). sum()/(len(df)))*100
missing_values_summary_after_imputation = pd.DataFrame({'total missings': df.isnull().sum(), 'percentage': 100 * df.isnull().sum() / len(df)})
print ('----------------------------------------------------')
print ('Updates data information')
print ('----------------------------------------------------')
print (missing_values_summary_after_imputation)
print('-------------------------------------------------------')

#%%
# One hot encoding 

from sklearn.preprocessing import OneHotEncoder

data_categorical = pd.DataFrame(df[categorical_cols], columns=categorical_cols)
one_hot_encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_categorical = one_hot_encoder.fit_transform(data_categorical)

encoded_categorical_df = pd.DataFrame(encoded_categorical, columns=one_hot_encoder.get_feature_names_out(categorical_cols))

data_encoded = pd.concat([df.drop(columns=categorical_cols), encoded_categorical_df], axis=1)

data_encoded.head()


#%%
# Summary statistics for the target variable
target_variable = 'AVERAGE DAYS TO ISSUE LICENSE'
mean_value = df[target_variable].mean()
median_value = df[target_variable].median()
std_value = df[target_variable].std()

# Create a DataFrame for the summary statistics
summary_data = {
    'Statistic': ['Mean', 'Median', 'Standard Deviation'],
    'Value': [f"{mean_value:.2f} days", f"{median_value:.2f} days", f"{std_value:.2f} days"]
}
summary_df = pd.DataFrame(summary_data)

# Display the summary table using pandas
print(summary_df)

# Save summary statistics as an image
import matplotlib.pyplot as plt
# Plot the histogram
plt.figure(figsize=(5, 3))
plt.hist(df['AVERAGE DAYS TO ISSUE LICENSE'].dropna(), bins=30, edgecolor='k', alpha=0.7)
plt.title('Distribution of Average Days to Issue License')
plt.xlabel('Average Days to Issue License')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
#%%
# data statistics 
Data_stat = df.describe().T
print ('--------------------------------------------------------------------------------')
print ('Data Summary')
print ('--------------------------------------------------------------------------------')
print  (Data_stat)
print ('--------------------------------------------------------------------------------')


#%%
import seaborn as sns
Data_stat_all = data_encoded.describe(include='all').T

# Define the number of rows and columns for the subplot grid
num_rows = 7
num_cols = 3
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20,35))
axes = axes.flatten()

# Plot each column in the DataFrame
#EDA UNIVARIATE 
for i, column in enumerate(data_encoded.columns):
    ax = axes[i]
    if data_encoded[column].dtype == 'object':
        sns.countplot(y=column, data=data_encoded, ax=ax, order=data_encoded[column].value_counts().index)
        ax.set_title(f'Count of {column}')
        ax.set_xlabel('Frequency')
    else:
        sns.histplot(data_encoded[column], bins=20, kde=True, ax=ax)
        ax.set_title(f'Distribution of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('count')
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)    

# Remove any empty subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
#%%
#creating duplicate dataset 
df2 = df.copy()

#%%

#Univariate analysis , 
yearly_avg_days = df2.groupby('YEAR')['AVERAGE DAYS TO ISSUE LICENSE'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.plot(yearly_avg_days['YEAR'], yearly_avg_days['AVERAGE DAYS TO ISSUE LICENSE'], marker='o', linestyle='-', color='Red')
plt.title('Average Days to Issue License Over the Years')
plt.xlabel('Year')
plt.ylabel('Average Days to Issue License')
plt.grid(True)
plt.show()

#%%
# colour bar histogram 
# Define the number of rows and columns for the subplot grid
num_row     
#%%
# Final evaluation on the held-out 10% test set
y_test_pred = Linear_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with Linear regression")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t test_Data (%) \t\t  MSE  \t\t  R2Score \t\t  MAE  \t\t  RMSE ") 
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Print the results for the test set
print("\t\t\t {:.3f}".format(10) + "\t\t\t {:.3f}".format(test_mse) +
      "\t\t\t {:.3f}".format(test_r2) + "\t\t\t {:.3f}".format(test_mae) + "\t\t\t {:.3f}".format(test_rmse))

#%%
#K-Fold Cross-Validation
from sklearn.model_selection import KFold, cross_val_score, train_test_split

model = LinearRegression()

# Evaluate base performance using K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Perform K-Fold Cross-Validation on the training set
mse_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
rmse_scores = np.sqrt(-mse_scores)
mae_scores = cross_val_score(model, X_train, y_train, scoring='neg_mean_absolute_error', cv=kf)
r2_scores = cross_val_score(model, X_train, y_train, scoring='r2', cv=kf)

# Train the model on the entire training set
model.fit(X_train, y_train)

# Evaluate the model on the validation set
y_val_pred = model.predict(X_val)
val_mse = mean_squared_error(y_val, y_val_pred)
val_rmse = np.sqrt(val_mse)
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

# Print the K-Fold Cross-Validation results
print("-------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with K-Fold Cross Validation")
print("-------------------------------------------------------------------------")
print("\t Training Data (%) \t Validation Data (%) \t MSE \t RMSE \t MAE \t R2 Scores")
print("--------------------------------------------------------------------------------")

training_size = (kf.n_splits - 1) / kf.n_splits * 100
validation_size = 100 / kf.n_splits

print(f"\t {training_size:.3f} \t\t {validation_size:.3f} \t\t {-mse_scores.mean():.3f} \t {rmse_scores.mean():.3f} \t {-mae_scores.mean():.3f} \t {r2_scores.mean():.3f}")

# Print the validation set results
print("--------------------------------------------------------")
print("\t Final Validation Data Evaluation")
print("--------------------------------------------------------")
print(f"\t MSE: {val_mse:.3f}")
print(f"\t RMSE: {val_rmse:.3f}")
print(f"\t MAE: {val_mae:.3f}")
print(f"\t R2 Score: {val_r2:.3f}")

#%%
#K-fold cross validation 
# Evaluate on test set
y_test_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

# Print Test Results
print("Predicted with K-Fold Cross Validation Test Data Evaluation")
print("--------------------")
print(f"MSE: {mse_test:.3f}")
print(f"RMSE: {rmse_test:.3f}")
print(f"MAE: {mae_test:.3f}")
print(f"R2 Score: {r2_test:.3f}")

#%%
#polynomial features
from sklearn.preprocessing import PolynomialFeatures

# Generate polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_val_poly = poly.transform(X_val)
X_test_poly = poly.transform(X_test)

# Standardize the numerical features
scaler = StandardScaler()
X_train_poly = scaler.fit_transform(X_train_poly)
X_val_poly = scaler.transform(X_val_poly)
X_test_poly = scaler.transform(X_test_poly)

# Train the model on the training set
Linear_model = LinearRegression()
Linear_model.fit(X_train_poly, y_train)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print(" Predicted with Polynomial Features")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t\t Train_Data (%) \t\t\t Val_Data (%) \t\t\t  MSE  \t\t\t  R2Score \t\t\t MAE  \t\t\tRMSE")
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Predict on the validation set
y_val_pred_poly = Linear_model.predict(X_val_poly)

# Calculate the metrics on the validation set
mse_val = mean_squared_error(y_val, y_val_pred_poly)
mae_val = mean_absolute_error(y_val, y_val_pred_poly)
r2_val = r2_score(y_val, y_val_pred_poly)
rmse_val = np.sqrt(mse_val)

# Print the results for the validation set
print("\t\t\t {:.3f}".format(80) + "\t\t\t\t {:.3f}".format(20) +
      "\t\t\t\t {:.3f}".format(mse_val) + "\t\t\t\t {:.3f}".format(r2_val) + "\t\t\t\t {:.3f}".format(mae_val) + "\t\t\t\t {:.3f}".format(rmse_val))

# Plot the actual vs predicted targets for validation data
x = np.arange(y_val.shape[0])
plt.plot(x, y_val, label='Actual Targets')
plt.plot(x, y_val_pred, label='Predicted Targets')
plt.title('Linear Regression 20% Validation Data, MSE ' + str("{:.3f}".format(mse_val)))
plt.xlabel('Samples')
plt.ylabel('Average Days to Issue License')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Final evaluation on the held-out 10% test set
y_test_pred = Linear_model.predict(X_test_poly)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print(" Predicted with Polynomial Features")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t test_Data (%) \t\t  MSE  \t\t  R2Score \t\t  MAE  \t\t  RMSE ") 
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Print the results for the test set
print("\t\t\t {:.3f}".format(10) + "\t\t\t {:.3f}".format(test_mse) +
      "\t\t\t {:.3f}".format(test_r2) + "\t\t\t {:.3f}".format(test_mae) + "\t\t\t {:.3f}".format(test_rmse))

#%%
#Regularization for improvement od model performance 
#ridge model 
from sklearn.linear_model import Ridge
# Train the model on the training set
ridge_model = Ridge()
ridge_model.fit(X_train, y_train)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with Ridge regression")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t\t Train_Data (%) \t\t\t Val_Data (%) \t\t\t  MSE  \t\t\t  R2Score \t\t\t MAE  \t\t\tRMSE")
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Predict on the validation set
y_val_pred = ridge_model.predict(X_val)

# Calculate the metrics on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)

# Print the results for the validation set
print("\t\t\t {:.3f}".format(81) + "\t\t\t\t {:.3f}".format(9) +
      "\t\t\t\t {:.3f}".format(mse_val) + "\t\t\t\t {:.3f}".format(r2_val) + "\t\t\t\t {:.3f}".format(mae_val) + "\t\t\t\t {:.3f}".format(rmse_val))

# Plot the actual vs predicted targets for validation data
x = np.arange(y_val.shape[0])
plt.plot(x, y_val, label='Actual Targets')
plt.plot(x, y_val_pred, label='Predicted Targets')
plt.title('Linear Regression 9% Validation Data, MSE ' + str("{:.3f}".format(mse_val)))
plt.xlabel('Samples')
plt.ylabel('Total Licenses Issued')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Final evaluation on the held-out 10% test set
y_test_pred = ridge_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with Ridge regression")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t test_Data (%) \t\t  MSE  \t\t  R2Score \t\t  MAE  \t\t  RMSE")
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Print the results for the test set
# Print the results for the test set
print("\t\t\t {:.3f}".format(0.1) + "\t\t\t {:.3f}".format(test_mse) +
      "\t\t\t {:.3f}".format(test_r2) + "\t\t\t {:.3f}".format(test_mae)+"\t\t\t {:.3f}".format(test_rmse))

#%%
#lasso model 

from sklearn.linear_model import Lasso
# Train the model on the training set
Lasso_model = Lasso()
Lasso_model.fit(X_train, y_train)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with Lasso Regression")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t\t Train_Data (%) \t\t\t Val_Data (%) \t\t\t  MSE  \t\t\t  R2Score \t\t\t MAE  \t\t\tRMSE")
print("------------------------------------------------------------------------------------------------------------------------------------------")


# Predict on the validation set
y_val_pred = Lasso_model.predict(X_val)

# Calculate the metrics on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)

# Print the results for the validation set
print("\t\t\t {:.3f}".format(81) + "\t\t\t\t {:.3f}".format(9) +
      "\t\t\t\t {:.3f}".format(mse_val) + "\t\t\t\t {:.3f}".format(r2_val) + "\t\t\t\t {:.3f}".format(mae_val) + "\t\t\t\t {:.3f}".format(rmse_val))

# Plot the actual vs predicted targets for validation data
x = np.arange(y_val.shape[0])
plt.plot(x, y_val, label='Actual Targets')
plt.plot(x, y_val_pred, label='Predicted Targets')
plt.title('Linear Regression 9% Validation Data, MSE ' + str("{:.3f}".format(mse_val)))
plt.xlabel('Samples')
plt.ylabel('Total Licenses Issued')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Final evaluation on the held-out 10% test set
y_test_pred = Lasso_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with Lasso regression")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t test_Data (%) \t\t  MSE  \t\t  R2Score \t\t  MAE  \t\t  RMSE")
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Print the results for the test set
# Print the results for the test set
print("\t\t\t {:.3f}".format(0.1) + "\t\t\t {:.3f}".format(test_mse) +
      "\t\t\t {:.3f}".format(test_r2) + "\t\t\t {:.3f}".format(test_mae)+"\t\t\t {:.3f}".format(test_rmse))

# Plot the actual vs predicted targets for validation data
x = np.arange(y_test.shape[0])
plt.plot(x, y_test, label='Actual Targets')
plt.plot(x, y_test_pred, label='Predicted Targets')
plt.title('Lasso Regression 9% Validation Data, R2score ' + str("{:.3f}".format(test_r2)))
plt.xlabel('Samples')
plt.ylabel('Average days to issue license')
plt.legend()
plt.grid(True)
plt.show()

#%%
#Advanced Regression models 
#Randomforest Regression 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import matplotlib.pyplot as plt

# Define features and target variable
X = df_final.drop(columns=['AVERAGE DAYS TO ISSUE LICENSE'])
y = df_final['AVERAGE DAYS TO ISSUE LICENSE']

# Split the data into training+validation (90%) and test sets (10%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Split the training+validation into training (80%) and validation (10% of original data)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1111, random_state=42)

# Standardize the numerical features (if necessary)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with Linear regression")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\tTraining Data (%) \t Validation Data (%) \t\t Mean Squared Error (MSE) \t\t R2Score(r2) \t\t Mean Absolute Error(mae)")
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Predict on the validation set
y_val_pred = rf_model.predict(X_val)

# Calculate the metrics on the validation set
mse_val = mean_squared_error(y_val, y_val_pred)
mae_val = mean_absolute_error(y_val, y_val_pred)
r2_val = r2_score(y_val, y_val_pred)
rmse_val = np.sqrt(mse_val)

# Print the results for the validation set
print("\t\t\t {:.3f}".format(81) + "\t\t\t\t {:.3f}".format(9) +
      "\t\t\t\t {:.3f}".format(mse_val) + "\t\t\t\t {:.3f}".format(r2_val) + "\t\t\t\t {:.3f}".format(mae_val) + "\t\t\t\t {:.3f}".format(rmse_val))

# Plot the actual vs predicted targets for validation data
x = np.arange(y_val.shape[0])
plt.plot(x, y_val, label='Actual Targets')
plt.plot(x, y_val_pred, label='Predicted Targets')
plt.title('Random_Forest Regression 9% Validation Data, MSE ' + str("{:.3f}".format(mse_val)))
plt.xlabel('Samples')
plt.ylabel('Average days to issue license')
plt.legend()
plt.grid(True)
plt.show()

#%%
# Final evaluation on the held-out 10% test set
y_test_pred = rf_model.predict(X_test)
test_mse = mean_squared_error(y_test, y_test_pred)
test_r2 = r2_score(y_test, y_test_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(test_mse)

print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("All Attributes Selected & Targets Predicted with Lasso regression")
print("-----------------------------------------------------------------------------------------------------------------------------------------")
print("\t\t test_Data (%) \t\t  MSE  \t\t  R2Score \t\t  MAE  \t\t  RMSE")
print("------------------------------------------------------------------------------------------------------------------------------------------")

# Print the results for the test set
# Print the results for the test set
print("\t\t\t {:.3f}".format(0.1) + "\t\t\t {:.3f}".format(test_mse) +
      "\t\t\t {:.3f}".format(test_r2) + "\t\t\t {:.3f}".format(test_mae)+"\t\t\t {:.3f}".format(test_rmse))
# Plot the actual vs predicted targets for validation data
x = np.arange(y_test.shape[0])
plt.plot(x, y_test, label='Actual Targets')
plt.plot(x, y_test_pred, label='Predicted Targets')
plt.title('Random_Forest Regression 9% Validation Data, R2score ' + str("{:.3f}".format(test_r2)))
plt.xlabel('Samples')
plt.ylabel('Average days to issue license')
plt.legend()
plt.grid(True)
plt.show()
