
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    classification_report
)

# 1. Set project and dataset paths
week4_folder = Path(__file__).resolve().parent.parent
project_folder = week4_folder.parent

dataset_path = (
    project_folder
    / "Week-1-Data-Acquisition-Cleaning-EDA"
    / "dataset"
    / "Online Retail.xlsx"
)

visualization_folder = week4_folder / "visualizations"
visualization_folder.mkdir(parents=True, exist_ok=True)

# 2. Load the dataset
if not dataset_path.exists():
    raise FileNotFoundError(
        f"Dataset not found at: {dataset_path}"
    )

df = pd.read_excel(dataset_path)

print("Original dataset shape:", df.shape)

# 3. Clean the data
df = df.drop_duplicates()
df = df.dropna(subset=["Description", "InvoiceDate", "Country"])

df = df[
    (df["Quantity"] > 0) &
    (df["UnitPrice"] > 0)
].copy()

df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"], errors="coerce"
)
df = df.dropna(subset=["InvoiceDate"])

# 4. Create revenue and useful features
df["Revenue"] = df["Quantity"] * df["UnitPrice"]
df["Month"] = df["InvoiceDate"].dt.month

# Keep a manageable, reproducible sample
if len(df) > 100000:
    df = df.sample(n=100000, random_state=42)

print("Cleaned dataset shape:", df.shape)

# 5. Create the target variable
# 1 = revenue above the median
# 0 = revenue at or below the median
revenue_median = df["Revenue"].median()
df["HighRevenue"] = (
    df["Revenue"] > revenue_median
).astype(int)

print("Revenue threshold:", round(revenue_median, 2))
print("\nTarget class counts:")
print(df["HighRevenue"].value_counts())

# 6. Select input features and target
# Revenue and Quantity are excluded from the input features.
X = df[["UnitPrice", "Country", "Month"]]
y = df["HighRevenue"]

numeric_features = ["UnitPrice", "Month"]
categorical_features = ["Country"]

# 7. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining records:", len(X_train))
print("Testing records:", len(X_test))

# 8. Preprocess the features
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", StandardScaler(), numeric_features),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)

# 9. Build and train the Logistic Regression model
model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

model.fit(X_train, y_train)

# 10. Make predictions
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# 11. Evaluate model performance
print("\n========== MODEL EVALUATION ==========")

print("Training accuracy:",
      round(accuracy_score(y_train, y_train_pred), 4))

print("Testing accuracy:",
      round(accuracy_score(y_test, y_pred), 4))

print("Precision:",
      round(precision_score(y_test, y_pred, zero_division=0), 4))

print("Recall:",
      round(recall_score(y_test, y_pred, zero_division=0), 4))

print("F1-score:",
      round(f1_score(y_test, y_pred, zero_division=0), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, y_prob), 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# 12. Visualization 1: Confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Low/Median Revenue", "High Revenue"]
)

display.plot(cmap="Blues", values_format="d")
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()

plt.savefig(
    visualization_folder / "confusion_matrix.png",
    dpi=300
)
plt.show()

# 13. Visualization 2: ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
auc_score = roc_auc_score(y_test, y_prob)

plt.figure(figsize=(8, 6))
plt.plot(
    fpr,
    tpr,
    label=f"Logistic Regression (AUC = {auc_score:.3f})"
)
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    visualization_folder / "roc_curve.png",
    dpi=300
)
plt.show()

print("\nCharts saved in:", visualization_folder)
print("\nWeek 4 model development completed!")
