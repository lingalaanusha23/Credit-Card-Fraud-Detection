# 🔍 Credit Card Fraud Detection using Isolation Forest

![Python](https://img.shields.io/badge/Python-3.7+-blue) ![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange) ![Pandas](https://img.shields.io/badge/Pandas-1.3+-green) ![Matplotlib](https://img.shields.io/badge/Matplotlib-3.5+-yellow)

Anomaly detection system for identifying potential credit card fraud using Isolation Forest algorithm.

## 📊 Dataset Overview
- Source: [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Features: 30 numerical features (V1-V28) + Amount + Class
- Target: Binary classification (0=normal, 1=fraud)

## 🛠️ Dependencies
```bash
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
```

## 🚀 Quick Start
1. Download dataset from Kaggle
2. Place `creditcard.csv` in project directory
3. Run the script:
```bash
python credit card fraud detection.py
```

## 📋 Code Structure
```python
# 1. Data Loading & Exploration
credit_card_data = pd.read_csv('creditcard.csv')

# 2. Data Visualization
credit_card_data.hist(bins=50, figsize=(20,15))
credit_card_data[cols_to_plot].plot(kind='box')

# 3. Model Training
if_model = IsolationForest(contamination=0.01)
if_model.fit(credit_card_data)

# 4. Anomaly Detection
if_predictions = if_model.predict(credit_card_data)

# 5. Results Visualization
plt.scatter(credit_card_data.iloc[:,0], credit_card_data.iloc[:,1], c=if_predictions)
```

## 📈 Visualizations
1. **Feature Distributions**: Histograms of all numerical features
2. **Box Plots**: Visualize spread and outliers in feature groups
3. **Anomaly Plot**: 2D scatter plot showing detected outliers

## ⚙️ Configuration
```python
# Model Parameters
contamination = 0.01  # Expected proportion of outliers

# Visualization Settings
plots_per_figure = 8  # Number of boxplots per figure
```

## 📊 Results Interpretation
- `-1` = Outlier (potential fraud)
- `1` = Inlier (normal transaction)

## 📝 Key Findings
1. Dataset contains 284,807 transactions
2. Features V1-V28 are PCA transformed (for privacy)
3. Isolation Forest identifies unusual transaction patterns

## 🚧 Limitations
- No feature engineering performed
- Basic visualization approach
- No performance metrics calculated

## 📚 Future Improvements
- Add proper train/test split
- Implement performance metrics (precision, recall)
- Try other anomaly detection algorithms
- Add feature importance analysis

## Output

https://github.com/user-attachments/assets/6a8d2816-74a2-49c1-b97d-a9fd9b5bbfe5

