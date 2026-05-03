import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("TELCO CUSTOMER CHURN ANALYSIS")
print("=" * 60)

print("\n[1/8] Loading dataset...")
df = pd.read_csv('dataset.csv')
print(f"   Dataset loaded: {df.shape[0]} customers, {df.shape[1]} columns")

print("\n[2/8] Data Cleaning...")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
print("   - TotalCharges cleaned")

service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
numeric_services = pd.DataFrame()
for col in service_cols:
    if col in df.columns:
        numeric_services[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}).fillna(0).astype(int)
df['TotalServices'] = numeric_services.sum(axis=1)
print("   - TotalServices feature created")

df = df.drop('customerID', axis=1)
print("   - customerID removed")

le = LabelEncoder()
df['Churn'] = le.fit_transform(df['Churn'])
print("   - Churn encoded: Yes=1, No=0")

categorical_cols = df.select_dtypes(include='object').columns.tolist()
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print("   - Categorical columns one-hot encoded")

print("\n[3/8] Exploratory Data Analysis...")

churn_counts = df['Churn'].value_counts()
total_customers = len(df)
churned = churn_counts.get(1, 0)
retained = churn_counts.get(0, 0)
churn_rate = (churned / total_customers) * 100

print(f"\n   === KEY RESULTS ===")
print(f"   Total Customers: {total_customers}")
print(f"   Churned Customers: {churned} ({churn_rate:.1f}%)")
print(f"   Retained Customers: {retained} ({100-churn_rate:.1f}%)")

print("\n   Saving charts...")

fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#FF6347', '#32CD32']
ax.bar(['Churned', 'Retained'], [churned, retained], color=colors)
ax.set_title('Customer Churn Distribution', fontsize=14)
ax.set_ylabel('Number of Customers')
for i, v in enumerate([churned, retained]):
    ax.text(i, v + 50, str(v), ha='center', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/01_churn_distribution.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/01_churn_distribution.png")

df_temp = pd.read_csv('dataset.csv')

fig, ax = plt.subplots(figsize=(8, 5))
churn_colors = ['#32CD32', '#FF6347']
sns.countplot(data=df_temp, x='gender', hue='Churn', palette=churn_colors, ax=ax)
ax.set_title('Churn Distribution by Gender')
ax.set_xlabel('Gender')
ax.set_ylabel('Count')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/02_churn_by_gender.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/02_churn_by_gender.png")

fig, ax = plt.subplots(figsize=(8, 5))
contract_churn = df_temp.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
ax.bar(contract_churn.index, contract_churn.values, color=['skyblue', 'lightcoral', 'lightgreen'])
ax.set_title('Churn Rate by Contract Type', fontsize=12)
ax.set_ylabel('Churn Rate')
ax.set_ylim(0, 0.5)
for i, v in enumerate(contract_churn.values):
    ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/03_churn_by_contract.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/03_churn_by_contract.png")

fig, ax = plt.subplots(figsize=(8, 5))
internet_churn = df_temp.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean())
ax.bar(internet_churn.index, internet_churn.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
ax.set_title('Churn Rate by Internet Service', fontsize=12)
ax.set_ylabel('Churn Rate')
ax.set_ylim(0, 0.5)
for i, v in enumerate(internet_churn.values):
    ax.text(i, v + 0.01, f'{v:.1%}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/04_churn_by_internet.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/04_churn_by_internet.png")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x='Churn', y='tenure', data=df_temp, palette=churn_colors)
plt.title('Tenure vs Churn Status')
plt.xlabel('Churn')
plt.ylabel('Tenure (Months)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/05_tenure_churn.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/05_tenure_churn.png")

fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df_temp, palette=churn_colors)
plt.title('Monthly Charges vs Churn Status')
plt.xlabel('Churn')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/06_monthly_charges_churn.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/06_monthly_charges_churn.png")

print("\n[4/8] Preparing features for modeling...")
X = df.drop('Churn', axis=1)
y = df['Churn']

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices']
scaler = StandardScaler()
X[numerical_features] = scaler.fit_transform(X[numerical_features])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Testing set: {X_test.shape[0]} samples")

print("\n[5/8] Training Logistic Regression Model...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_preds = lr_model.predict(X_test)
lr_proba = lr_model.predict_proba(X_test)[:, 1]

lr_accuracy = accuracy_score(y_test, lr_preds)
lr_precision = precision_score(y_test, lr_preds)
lr_recall = recall_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds)
lr_roc = roc_auc_score(y_test, lr_proba)

print("\n   === LOGISTIC REGRESSION PERFORMANCE ===")
print(f"   Accuracy:  {lr_accuracy:.2%}")
print(f"   Precision: {lr_precision:.2%}")
print(f"   Recall:    {lr_recall:.2%}")
print(f"   F1-Score:  {lr_f1:.2%}")
print(f"   ROC AUC:   {lr_roc:.2%}")

print("\n[6/8] Training XGBoost Model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_proba = xgb_model.predict_proba(X_test)[:, 1]

xgb_accuracy = accuracy_score(y_test, xgb_preds)
xgb_precision = precision_score(y_test, xgb_preds)
xgb_recall = recall_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_roc = roc_auc_score(y_test, xgb_proba)

print("\n   === XGBOOST PERFORMANCE ===")
print(f"   Accuracy:  {xgb_accuracy:.2%}")
print(f"   Precision: {xgb_precision:.2%}")
print(f"   Recall:    {xgb_recall:.2%}")
print(f"   F1-Score:  {xgb_f1:.2%}")
print(f"   ROC AUC:   {xgb_roc:.2%}")

print("\n[7/8] Model Comparison...")
print("\n   === MODEL COMPARISON ===")
print(f"   {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC AUC':<12}")
print(f"   {'-'*75}")
print(f"   {'Logistic Regression':<25} {lr_accuracy:.2%}       {lr_precision:.2%}       {lr_recall:.2%}       {lr_f1:.2%}       {lr_roc:.2%}")
print(f"   {'XGBoost':<25} {xgb_accuracy:.2%}       {xgb_precision:.2%}       {xgb_recall:.2%}       {xgb_f1:.2%}       {xgb_roc:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_lr = confusion_matrix(y_test, lr_preds)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[0].set_title('Logistic Regression\nConfusion Matrix')

cm_xgb = confusion_matrix(y_test, xgb_preds)
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
axes[1].set_title('XGBoost\nConfusion Matrix')

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/07_confusion_matrices.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/07_confusion_matrices.png")

fig, ax = plt.subplots(figsize=(10, 6))
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=True).tail(15)

ax.barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
ax.set_title('Top 15 Features - XGBoost Feature Importance', fontsize=12)
ax.set_xlabel('Importance')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/08_feature_importance.png', dpi=150)
plt.close()
print(f"   - {OUTPUT_DIR}/08_feature_importance.png")

print("\n[8/8] Business Insights...")

print("\n   === TOP CHURN RISK FACTORS ===")
feature_imp = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

top_factors = feature_imp.head(5)
for idx, row in top_factors.iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

print("\n   === BUSINESS RECOMMENDATIONS ===")
print("   1. Contract Type: Focus on month-to-month customers - offer incentives for longer contracts")
print("   2. Tenure: Implement early retention programs for new customers (first 12 months)")
print("   3. Internet Service: Investigate fiber optic service quality - high churn observed")
print("   4. Monthly Charges: Review pricing strategy for high-paying customers")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)
print(f"\nAll charts saved to: {OUTPUT_DIR}/")
print("\n" + "=" * 60)
print("KEY TAKEAWAYS FOR PRESENTATION")
print("=" * 60)
print(f"""
Dataset Overview:
  - Total Customers: {total_customers:,}
  - Churned: {churned:,} ({churn_rate:.1f}%)
  - Retained: {retained:,} ({100-churn_rate:.1f}%)

Model Performance:
  - Logistic Regression Accuracy: {lr_accuracy:.1%}
  - XGBoost Accuracy: {xgb_accuracy:.1%}

Top Churn Risk Factors:
  - Contract Type (Month-to-month has highest churn)
  - Tenure (New customers more likely to churn)
  - Internet Service (Fiber optic customers at risk)
""")
print("=" * 60)