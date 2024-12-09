# Evaluation report

## code snippet of best and last model:

![image](https://github.com/user-attachments/assets/6dda2f70-994f-4f8b-b738-7c5e3b7522ca)

## MAE-test = 77064

## RMSE-test = 115704

## R²-test = 0.72

## Feature selection:

![image](https://github.com/user-attachments/assets/6cf2b845-5c94-43ae-8081-7199649784e1)

## Accuracy computing procedure:

The data is split into 20% test set and 80% training with a 5-fold cross validation.

## Model efficiency: 

- training time = 0.6s
- inference time = 1.5s

## Final dataset:

- The dataset contained originally 24397 records.
- Outliers were removed with 3-sigma rule.
- Entries of which municipalitites contained less then 20 properties were removed.
- Finally 5990 records remained.


















