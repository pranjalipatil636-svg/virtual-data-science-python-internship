
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Load the Online Retail dataset
df = pd.read_excel(
    "Week-1-Data-Acquisition-Cleaning-EDA/dataset/Online Retail.xlsx"
)

print("Original dataset shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates()

# Remove rows with missing product descriptions
df = df.dropna(subset=["Description"])

# Keep positive sales transactions
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Calculate transaction revenue
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Select the United Kingdom and Germany
uk_revenue = df.loc[df["Country"] == "United Kingdom", "Revenue"]
germany_revenue = df.loc[df["Country"] == "Germany", "Revenue"]

print("\nUnited Kingdom transactions:", len(uk_revenue))
print("Germany transactions:", len(germany_revenue))

print("\nMean revenue - United Kingdom:", round(uk_revenue.mean(), 2))
print("Mean revenue - Germany:", round(germany_revenue.mean(), 2))

# Welch's independent two-sample t-test
t_stat, p_value = stats.ttest_ind(
    uk_revenue,
    germany_revenue,
    equal_var=False
)

print("\nT-statistic:", round(t_stat, 4))
print("P-value:", p_value)

# Interpret the result
alpha = 0.05

if p_value < alpha:
    print("Decision: Reject the null hypothesis.")
    print("Conclusion: The mean transaction revenues differ statistically.")
else:
    print("Decision: Fail to reject the null hypothesis.")
    print("Conclusion: There is insufficient evidence of a difference.")

# Visualization 1: Compare mean transaction revenue

mean_revenue = pd.Series({
    "United Kingdom": uk_revenue.mean(),
    "Germany": germany_revenue.mean()
})

plt.figure(figsize=(8, 5))
bars = plt.bar(
    mean_revenue.index,
    mean_revenue.values
)

plt.title("Average Transaction Revenue: UK vs Germany")
plt.xlabel("Country")
plt.ylabel("Average Revenue")

# Display values above the bars
plt.bar_label(bars, fmt="%.2f", padding=3)

plt.tight_layout()
plt.savefig(
    "Week-3-Statistical-Analysis/visualizations/mean_revenue_comparison.png",
    dpi=300
)
plt.show()

# Visualization 2: Revenue distribution by country

plt.figure(figsize=(9, 6))

plt.boxplot(
    [uk_revenue, germany_revenue],
    tick_labels=["United Kingdom", "Germany"],
    showfliers=False
)

plt.yscale("log")
plt.title("Transaction Revenue Distribution: UK vs Germany")
plt.xlabel("Country")
plt.ylabel("Transaction Revenue (log scale)")
plt.tight_layout()

plt.savefig(
    "Week-3-Statistical-Analysis/visualizations/revenue_distribution_by_country.png",
    dpi=300
)

plt.show()

# Step 6: Calculate the 95% confidence interval
import numpy as np

uk_mean = uk_revenue.mean()
germany_mean = germany_revenue.mean()

mean_difference = uk_mean - germany_mean

# Standard error using Welch's method
uk_var = uk_revenue.var(ddof=1)
germany_var = germany_revenue.var(ddof=1)

uk_n = len(uk_revenue)
germany_n = len(germany_revenue)

standard_error = np.sqrt(
    uk_var / uk_n + germany_var / germany_n
)

# Welch-Satterthwaite degrees of freedom
numerator = (uk_var / uk_n + germany_var / germany_n) ** 2

denominator = (
    (uk_var / uk_n) ** 2 / (uk_n - 1)
    + (germany_var / germany_n) ** 2 / (germany_n - 1)
)

degrees_freedom = numerator / denominator

# 95% confidence interval
critical_value = stats.t.ppf(0.975, degrees_freedom)

lower_bound = mean_difference - critical_value * standard_error
upper_bound = mean_difference + critical_value * standard_error

print("\n========== 95% CONFIDENCE INTERVAL ==========")
print("Mean difference (UK - Germany):", round(mean_difference, 2))
print("95% CI lower bound:", round(lower_bound, 2))
print("95% CI upper bound:", round(upper_bound, 2))
