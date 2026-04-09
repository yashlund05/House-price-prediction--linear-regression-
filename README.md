# 📈 Numerical Methods Linear Regression (From Scratch)

This project implements **Simple Linear Regression using pure numerical methods and statistical formulas** without using any machine learning libraries such as scikit-learn.

All calculations such as **mean, median, mode, variance, standard deviation, correlation (r), regression coefficients (m, c), error metrics, and plots** are computed manually using formulas from Numerical Methods and Statistics.

---

## 🎯 Objective

To demonstrate how Linear Regression works internally by computing every component using:

- Summations
- Deviations from mean
- Pearson correlation formula
- Numerical error calculations
- Graphical visualization

---

## 📂 Dataset

**File:** `Housing.csv`  
**Target Variable (Y):** Price  
**Input Variable (X):** Avg. Area Income

The dataset contains housing data including income, house age, rooms, bedrooms, population, and price.

---

## 🧹 Data Cleaning

- Dropped non-numeric column: `Address`
- Removed outliers using **Z-score method**
- Condition used: \(|Z| > 3\)

---

## 📊 Statistical Measures Computed

- Mean
- Median
- Mode (via binning)
- Variance
- Standard Deviation
- Deviation from Mean

---

## 🔗 Correlation and Regression (Using Formulas)

- Pearson Correlation Coefficient \( r \)
- Slope \( m \)
- Intercept \( c \)
- Regression Equation: \( \hat{Y} = mX + c \)

---

## 📉 Error Metrics Calculated

- Residuals
- Absolute Error
- Squared Error
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE (Root Mean Squared Error)
- \( R^2 \) (Coefficient of Determination)

---

## 📈 Plots Generated

The script generates and saves the following plots into:

```
regression_output.png
```

1. Scatter plot with regression line
2. Residual plot
3. Error distribution histogram
4. Regression visualization

---

## 🛠️ Tools & Libraries Used

- Python
- pandas
- numpy
- matplotlib

---

## ▶️ How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset

Put `Housing.csv` in the same folder as the script.

### 3. Run the program

```bash
python main.py
```

---

## ✅ What Makes This Project Different

- No ML libraries used
- Entire regression built from scratch
- Pure Numerical Methods implementation
- Educational demonstration of how regression works internally

---

## 📌 Output

- Console output showing all statistical calculations
- `regression_output.png` with all graphs

---

## 👨‍🎓 Academic Relevance

This project is ideal for subjects like:

- Numerical Methods
- Statistics
- Data Analysis Fundamentals

It clearly shows the mathematical foundation behind Linear Regression..
