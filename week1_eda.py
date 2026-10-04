import pandas as pd

df = pd.read_excel("Week-1-Data-Acquisition-Cleaning-EDA/dataset/Online Retail.xlsx")

print(df.head())
print("\nShape of dataset:")
print(df.shape)

print("\nColumn names:")
print(df.columns)

print("\nDataset information:")
print(df.info())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:")
print(df.duplicated().sum())
df = df.drop_duplicates()

print("\nShape after removing duplicates:")
print(df.shape)
print("\nRows with missing CustomerID:")
print(df[df["CustomerID"].isnull()].head())
print("\nMissing value percentage:")
print((df.isnull().sum() / len(df) * 100).round(2))
print("\nRows with missing Description:")
print(df[df["Description"].isnull()].head())
print("\nRows with missing Description and their Quantity:")
print(df[df["Description"].isnull()][["InvoiceNo", "StockCode", "Quantity", "UnitPrice", "Country"]].head(10))

print("\nQuantity statistics for missing Description:")
print(df[df["Description"].isnull()]["Quantity"].describe())
print("\nRemoving rows with missing Description...")

df = df.dropna(subset=["Description"])

print("Shape after removing missing Description:")
print(df.shape)

print("\nRemaining missing values:")
print(df.isnull().sum())
print("\nRows with zero or negative Quantity:")
print(df[df["Quantity"] <= 0].head())

print("\nNumber of rows with zero or negative Quantity:")
print((df["Quantity"] <= 0).sum())

print("\nRows with zero or negative UnitPrice:")
print(df[df["UnitPrice"] <= 0].head())

print("\nNumber of rows with zero or negative UnitPrice:")
print((df["UnitPrice"] <= 0).sum())
print("\nRows with Quantity <= 0:")
print((df["Quantity"] <= 0).sum())

print("\nRows with UnitPrice <= 0:")
print((df["UnitPrice"] <= 0).sum())

print("\nRows satisfying both conditions:")
print(((df["Quantity"] <= 0) & (df["UnitPrice"] <= 0)).sum())

print("\nRows with valid sales values:")
print(((df["Quantity"] > 0) & (df["UnitPrice"] > 0)).sum())
print("\nCreating clean sales dataset...")

df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

print("Final shape of clean dataset:")
print(df.shape)

print("\nFinal missing values:")
print(df.isnull().sum())
df["Revenue"] = df["Quantity"] * df["UnitPrice"]

print("\nRevenue column added.")

print("\nRevenue statistics:")
print(df["Revenue"].describe())
print("\nSummary statistics:")
print(df[["Quantity", "UnitPrice", "Revenue"]].describe())
top_products = (
    df.groupby("Description")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 products by revenue:")
print(top_products)
top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 countries by revenue:")
print(top_countries)
import matplotlib.pyplot as plt
# Top 10 countries by revenue

top_countries = (
    df.groupby("Country")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10, 6))
top_countries.plot(kind="bar")

plt.title("Top 10 Countries by Revenue")
plt.xlabel("Country")
plt.ylabel("Total Revenue")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "Week-1-Data-Acquisition-Cleaning-EDA/visualizations/top_10_countries_revenue.png"
)

plt.show()
# Monthly Revenue Trend

df["Month"] = df["InvoiceDate"].dt.to_period("M")

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
    "Week-1-Data-Acquisition-Cleaning-EDA/visualizations/monthly_revenue_trend.png"
)

plt.show()
# Top 10 Products by Revenue

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
    "Week-1-Data-Acquisition-Cleaning-EDA/visualizations/top_10_products_revenue.png"
)

plt.show()
# Final EDA Summary

total_revenue = df["Revenue"].sum()
total_quantity = df["Quantity"].sum()
total_products = df["Description"].nunique()
total_countries = df["Country"].nunique()

best_country = (
    df.groupby("Country")["Revenue"]
    .sum()
    .idxmax()
)

best_product = (
    df.groupby("Description")["Revenue"]
    .sum()
    .idxmax()
)

best_month = (
    df.groupby("Month")["Revenue"]
    .sum()
    .idxmax()
)

print("\n========== FINAL EDA SUMMARY ==========")

print("Total Revenue:", round(total_revenue, 2))
print("Total Quantity Sold:", total_quantity)
print("Number of Unique Products:", total_products)
print("Number of Countries:", total_countries)
print("Highest Revenue Country:", best_country)
print("Highest Revenue Product:", best_product)
print("Highest Revenue Month:", best_month)