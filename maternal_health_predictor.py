
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_validate, KFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

# Load dataset
df = pd.read_csv("/Users/mer/Downloads/Maternal Health Risk Data Set.csv")

# Explore the dataset
print("Shape:", df.shape)
print(df.info())
print(df.describe())
print("RiskLevel Distribution:")
print(df["RiskLevel"].value_counts())

# Visualize RiskLevel distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='RiskLevel', data=df, palette='Set2')
plt.title("Distribution of Maternal Health Risk Levels")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("risk_level_distribution.png")
plt.show()

# Boxplot of key features by Risk Level
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BS', 'BodyTemp', 'HeartRate']
for col in features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='RiskLevel', y=col, data=df, palette='pastel')
    plt.title(f"{col} by Risk Level")
    plt.tight_layout()
    plt.savefig(f"{col.lower()}_by_risk_level.png")
    plt.show()

# Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df.drop(columns='RiskLevel').corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("feature_correlation_heatmap.png")
plt.show()

# Encode RiskLevel
df["RiskLevel"] = df["RiskLevel"].str.strip().str.lower()
le = LabelEncoder()
df["RiskLevel_encoded"] = le.fit_transform(df["RiskLevel"])
print("Encoded labels:", dict(zip(le.classes_, le.transform(le.classes_))))

# Split data
X = df.drop(["RiskLevel", "RiskLevel_encoded"], axis=1)
y = df["RiskLevel_encoded"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
log_model = LogisticRegression(multi_class='ovr', max_iter=1000)
log_model.fit(X_train_scaled, y_train)
y_pred = log_model.predict(X_test_scaled)

# Evaluate model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Plot classification report metrics
report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose().iloc[:-3]

plt.figure(figsize=(8, 5))
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', colormap='Set2')
plt.title("Classification Metrics by Risk Level")
plt.xlabel("Risk Level")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.legend(loc='lower right')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig("classification_metrics_barplot.png")
plt.show()

