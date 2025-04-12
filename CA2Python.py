# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# 2. Load Dataset
df = pd.read_csv("household (3).csv")

# 3. Dataset Info
print(df.info())
print(df.head())

# 4. Handle Missing Data
df['exp_pw'].fillna(df['exp_pw'].mean(), inplace=True)
df['eqv_exp_pw'].fillna(df['eqv_exp_pw'].mean(), inplace=True)

# 5. Statistical Summary
print(df.describe())

# 6. Correlation Matrix
corr = df[['weight', 'exp_pw', 'eqv_exp_pw']].corr()

# 7. Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# 8. Boxplot for Outliers
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.boxplot(x=df['exp_pw'])
plt.title("Boxplot of exp_pw")

plt.subplot(1, 2, 2)
sns.boxplot(x=df['eqv_exp_pw'])
plt.title("Boxplot of eqv_exp_pw")
plt.tight_layout()
plt.show()

# 9. Outlier Detection with Z-Score
z_scores = np.abs(zscore(df[['exp_pw', 'eqv_exp_pw']]))
outliers = (z_scores > 3).sum(axis=0)
print("Outliers detected using Z-score:")
print(outliers)

# 10. Scatter Plot
plt.figure(figsize=(6, 4))
sns.scatterplot(data=df, x='exp_pw', y='eqv_exp_pw')
plt.title("Scatter Plot: exp_pw vs eqv_exp_pw")
plt.show()

# 11. Line Plot (over years)
avg_yearly = df.groupby('year')[['exp_pw', 'eqv_exp_pw']].mean()
avg_yearly.plot(marker='o', figsize=(8, 5))
plt.title("Average Expenditure Over Years")
plt.ylabel("Expenditure")
plt.grid(True)
plt.show()

# 12. Bar Plot
plt.figure(figsize=(8, 5))
sns.barplot(x='year', y='exp_pw', data=df.groupby('year').mean().reset_index())
plt.title("Mean exp_pw per Year")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 13. Column Graph (same as barplot but vertical grouping)
plt.figure(figsize=(8, 5))
sns.barplot(x='nzhec_short', y='exp_pw', data=df)
plt.title("exp_pw by Category")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# 14. Pair Plot
sns.pairplot(df[['weight', 'exp_pw', 'eqv_exp_pw']])
plt.suptitle("Pair Plot of Numerical Features", y=1.02)
plt.show()
