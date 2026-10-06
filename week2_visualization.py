import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_excel(
    "Week-1-Data-Acquisition-Cleaning-EDA/dataset/Online Retail.xlsx"
)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print(df.head())
# Remove duplicate rows
df = df.drop_duplicates()

# Remove rows with missing Description
df = df.dropna(subset=["Description"])

# Keep only valid sales records
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# Create Revenue column
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

# Create Month column
df["Month"] = df["InvoiceDate"].dt.to_period("M")

print("\nData prepared for visualization")
print("Cleaned shape:", df.shape)
# Visualization 1: Monthly Revenue Trend

monthly_revenue = (
    df.groupby("Month")["Revenue"]
    .sum()
)

plt.figure(figsize=(12, 6))
monthly_revenue.plot(kind="line", marker="o")

plt.title("Monthly Revenue Trend")
plt.xlabel("Month")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "Week-2-Advanced-Visualization/visualizations/monthly_revenue_trend.png"
)

plt.show()
# Visualization 2: Top 10 Countries by Revenue

top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
top_countries.sort_values().plot(kind="barh")

plt.title("Top 10 Countries by Revenue")
plt.xlabel("Total Revenue")
plt.ylabel("Country")
plt.tight_layout()

plt.savefig(
    "Week-2-Advanced-Visualization/visualizations/top_10_countries_revenue.png"
)

plt.show()
# Visualization 3: Top 10 Products by Revenue

top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))
top_products.sort_values().plot(kind="barh")

plt.title("Top 10 Products by Revenue")
plt.xlabel("Total Revenue")
plt.ylabel("Product")
plt.tight_layout()

plt.savefig(
    "Week-2-Advanced-Visualization/visualizations/top_10_products_revenue.png"
)

plt.show()
# Visualization 4: Revenue Distribution

plt.figure(figsize=(10, 6))

plt.hist(df["Revenue"], bins=50)

plt.title("Distribution of Transaction Revenue")
plt.xlabel("Revenue")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    "Week-2-Advanced-Visualization/visualizations/revenue_distribution.png"
)

plt.show()
# Visualization 5: Quantity vs Revenue

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Quantity"],
    df["Revenue"],
    alpha=0.3
)

plt.title("Quantity vs Revenue")
plt.xlabel("Quantity")
plt.ylabel("Revenue")

plt.tight_layout()

plt.savefig(
    "Week-2-Advanced-Visualization/visualizations/quantity_vs_revenue.png"
)

plt.show()
# Week 2 Key Insights

print("\n========== WEEK 2 KEY INSIGHTS ==========")

print("\nTotal Revenue:")
print(round(df["Revenue"].sum(), 2))

print("\nAverage Transaction Revenue:")
print(round(df["Revenue"].mean(), 2))

print("\nHighest Revenue Country:")
print(df.groupby("Country")["Revenue"].sum().idxmax())

print("\nHighest Revenue Month:")
print(df.groupby("Month")["Revenue"].sum().idxmax())

print("\nHighest Revenue Product:")
print(df.groupby("Description")["Revenue"].sum().idxmax())

print("\nAverage Quantity per Transaction:")
print(round(df["Quantity"].mean(), 2))