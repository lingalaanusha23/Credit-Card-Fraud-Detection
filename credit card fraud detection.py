import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

credit_card_data = pd.read_csv(r'/kaggle/input/creditcardfraud/creditcard.csv')

print("Dataset Head:")
print(credit_card_data.head())

print("\nDataset Shape:")
print(credit_card_data.shape)

print("\nMissing Values Count:")
print(credit_card_data.isnull().sum())

print("\nSummary Statistics:")
print(credit_card_data.describe())

credit_card_data.hist(bins=50, figsize=(20, 15))
plt.title('Distribution of Numerical Features')
plt.show()


num_columns = len(credit_card_data.columns)
plots_per_figure = 8
for i in range(0, num_columns, plots_per_figure):
    cols_to_plot = credit_card_data.columns[i:i + plots_per_figure]
    plt.figure(figsize=(20, 10))
    plt.title(f'Box Plots of Numerical Features (Columns {i+1} to {min(i + plots_per_figure, num_columns)})')
    credit_card_data[cols_to_plot].plot(kind='box', subplots=True, layout=(2, 4), figsize=(20,10))
    plt.tight_layout()
    plt.show()


if_model = IsolationForest(contamination=0.01)
if_model.fit(credit_card_data)
if_predictions = if_model.predict(credit_card_data)

outlier_data = pd.DataFrame({'outlier': if_predictions}, index=credit_card_data.index)
print("\nOutlier Predictions:")
print(outlier_data[outlier_data['outlier'] == -1].head())  # -1 indicates outliers


plt.figure(figsize=(10, 6))
plt.scatter(credit_card_data.iloc[:, 0], credit_card_data.iloc[:, 1],
             c=['red' if prediction == -1 else 'blue' for prediction in if_predictions])
plt.title('Outlier Visualization')
plt.xlabel(credit_card_data.columns[0])
plt.ylabel(credit_card_data.columns[1])
plt.show()