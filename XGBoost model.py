# Installing python modules

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

"""
Step 1 : Data cleaning

You have collected your data, you have cleaned and analyzed it a first time!
So it's time to do some machine learning with it!

But first, we have to prepare the data for machine learning.

- No duplicates.
- No NANs.
- No text data.
- No features that have too strong correlation between them.

"""

# 1.1 Loading real estate data into pandas dataframe

df=pd.read_csv("C:\\Users\\vande\\challenge-regression\\data\\real_estate_belgium.csv")

"""
Step 2: Features engineering

Select the best features in order to train your model. Don't forget you can add value and menaing by creating new features from your dataset. Use Open Data and other resources.

"""

# 2.1 Keep column features which have high correlation coefficients
df = df[["Municipality", "Price", "Living_Area", "Number_of_Rooms", "Fully_Equipped_Kitchen", "Terrace_Area","Garden_Area","Type_of_Property"]]

"""
Step 3: Data formatting

Now that the dataset is ready, you have to format it for machine learning:

- Divide your dataset for training and testing. (`X_train, y_train, X_test, y_test`)
"""

# 3.1 Extract feature and target arrays
X, y = df.drop('Price', axis=1), df[['Price']]

# 3.2 Extract text features
cats = X.select_dtypes(exclude=np.number).columns.tolist()

# 3.3 Convert to Pandas category
for col in cats:
   X[col] = X[col].astype('category')
X.dtypes

# 3.4 split dataset into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=1)

"""
Step 4: Model selection --> XGBoost

The dataset is ready. Now let's select a model, keep it to the model family you've been assigned.

Step 5: Apply your model

Apply your model on your data.
"""

# 5.1 Create XGBoost regression matrices

dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)


# 5.2 Define hyperparameters

params = {"objective": "reg:squarederror", "tree_method": "hist", "device":"cuda"}

# 5.3 Train model

n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)

# 5.4 Plot the feature importances

xgb.plot_importance(model)
plt.show()

# 5.5 Prediction

preds = model.predict(dtest_reg)

"""
Step 6: Model evaluation

Let's evaluate your model. Which accuracy did you reach? --> R-squared: 0.72, RMSE: 116639, MAE: 77739 for test data in base model.

Try to answer those questions:

- How could you improve this result? --> cross-validation 
- Which part of the process has the most impact on the results? --> feature selection
- How should you divide your time working on this kind of project?
- Try to illustrate your model if it's possible to make it interpretable.

"""
# 6.1 Model evaluation with r², rmse (root mean squared error), and mae (mean absolute error)

rmse = root_mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"R-squared: {r2:.2f}, RMSE: {rmse:.0f}, MAE: {mae:.0f} for test data in base model.")

# --> R-squared: 0.72, RMSE: 116639, MAE: 77739 for test data in base model.

# 6.2 Scatterplot of test and prediction set

plt.scatter(X_test['Terrace_Area'], y_test)
plt.scatter(X_test['Terrace_Area'], preds)
plt

# 6.3 Validation Sets During Training

# set up the parameters

params = {"objective": "reg:squarederror", "tree_method": "hist", "device":"cuda"}
n = 100

# setup list of two tuples that each contain two elements. 
# The first element is the array for the model to evaluate, and the second is the array’s name.

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

# pass this array to the evals parameter of xgb.train, we will see the model performance after each boosting round:

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=10 # Every ten rounds
)

## XGBoost early stopping

"""
Generally, the more rounds there are, the more XGBoost tries to minimize the loss. 
But this doesn’t mean the loss will always go down. Let’s try with 5000 boosting rounds with the verbosity of 500:

"""
params = {"objective": "reg:squarederror", "tree_method": "hist", "device":"cuda"}
n = 5000

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=250 # Every ten rounds
)

"""
We want the golden middle: a model that learned just enough patterns in training that it gives the highest performance on the validation set. So, how do we find the perfect number of boosting rounds, then?
We will use a technique called early stopping. Early stopping forces XGBoost to watch the validation loss, and if it stops improving for a specified number of rounds, it automatically stops training.
This means we can set as high a number of boosting rounds as long as we set a sensible number of early stopping rounds.

"""

params = {"objective": "reg:squarederror", "tree_method": "hist", "device":"cuda"}
n = 10000

evals = [(dtrain_reg, "train"), (dtest_reg, "validation")]

model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
   verbose_eval=10,
   # Activate early stopping
   early_stopping_rounds = 50

)

## XGBoost Cross-Validation

"""
Since we try to find the best value of a hyperparameter by comparing the validation performance of the model on the test set, 
we will end up with a model that is configured to perform well only on that particular test set. 

Instead, we want a model that performs well across the board — on any test set we throw at it.

A possible workaround is splitting the data into three sets. The model trains on the first set, the second set is used for evaluation and hyperparameter tuning, and the third is the final one we test the model before production.

But when data is limited, splitting data into three sets will make the training set sparse, which hurts model performance.

The solution to all these problems is cross-validation. In cross-validation, we still have two sets: training and testing.
While the test set waits in the corner, we split the training into 3, 5, 7, or k splits or folds. 
Then, we train the model k times. Each time, we use k-1 parts for training and the final kth part for validation. 
This process is called k-fold cross-validation:

"""

params = {"objective": "reg:squarederror", "tree_method": "hist", "device":"cuda"}
n = 1000

results = xgb.cv(
   params, dtrain_reg,
   num_boost_round=n, 
   nfold=5, # specify the number of splits
   early_stopping_rounds=20
)

# It has the same number of rows as the number of boosting rounds. 
# Each row is the average of all splits for that round. So, to find the best score, we take the minimum of the test-rmse-mean column:

best_rmse = results['test-rmse-mean'].min()
print(best_rmse)

# model prediction

preds = model.predict(dtest_reg)

# model evaluation

rmse = root_mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)
mae = mean_absolute_error(y_test, preds)
print(f"R-squared: {r2:.2f}, RMSE: {rmse:.0f}, MAE: {mae:.0f} for test data in base model.")

# --> R-squared: 0.72, RMSE: 115704, MAE: 77064 for test data in base model.

# plot tree model

xgb.plot_tree(model)
fig = plt.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree.png')
plt.show()