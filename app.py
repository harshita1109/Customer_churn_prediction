import streamlit as st
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

st.set_page_config(page_title="Telco Customer Churn Analysis", page_icon="📊", layout="wide")

@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('dataset.csv')
    
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    numeric_services = pd.DataFrame()
    for col in service_cols:
        if col in df.columns:
            numeric_services[col] = df[col].map({'Yes': 1, 'No': 0, 'No phone service': 0, 'No internet service': 0}).fillna(0).astype(int)
    df['TotalServices'] = numeric_services.sum(axis=1)
    
    df_processed = df.drop('customerID', axis=1)
    le = LabelEncoder()
    df_processed['Churn'] = le.fit_transform(df_processed['Churn'])
    categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df, df_processed

def train_models(df_processed):
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices']
    scaler = StandardScaler()
    X[numerical_features] = scaler.fit_transform(X[numerical_features])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_proba = lr_model.predict_proba(X_test)[:, 1]
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    xgb_preds = xgb_model.predict(X_test)
    xgb_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    return {
        'lr': {'model': lr_model, 'preds': lr_preds, 'proba': lr_proba, 'accuracy': accuracy_score(y_test, lr_preds), 'precision': precision_score(y_test, lr_preds), 'recall': recall_score(y_test, lr_preds), 'f1': f1_score(y_test, lr_preds), 'roc': roc_auc_score(y_test, lr_proba), 'y_test': y_test},
        'xgb': {'model': xgb_model, 'preds': xgb_preds, 'proba': xgb_proba, 'accuracy': accuracy_score(y_test, xgb_preds), 'precision': precision_score(y_test, xgb_preds), 'recall': recall_score(y_test, xgb_preds), 'f1': f1_score(y_test, xgb_preds), 'roc': roc_auc_score(y_test, xgb_proba), 'y_test': y_test, 'X': X},
        'X_test': X_test, 'y_test': y_test, 'X': X
    }

st.title("📊 Telco Customer Churn Analysis")
st.markdown("---")

df, df_processed = load_and_prepare_data()
models = train_models(df_processed)

total_customers = len(df)
churned = (df['Churn'] == 'Yes').sum()
retained = (df['Churn'] == 'No').sum()
churn_rate = (churned / total_customers) * 100

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Dashboard", "📈 EDA", "🤖 Models", "💡 Insights"])

with tab1:
    st.header("Customer Churn Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("Churned Customers", f"{churned:,}", delta_color="inverse")
    col3.metric("Retained Customers", f"{retained:,}")
    col4.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"-{churn_rate:.1f}%", delta_color="inverse")
    
    st.markdown("###")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ['#32CD32', '#FF6347']
        bars = ax.bar(['Retained', 'Churned'], [retained, churned], color=colors)
        ax.set_title('Customer Distribution', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Customers', fontsize=11)
        for bar, val in zip(bars, [retained, churned]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val:,}', ha='center', fontsize=12, fontweight='bold')
        ax.set_ylim(0, max(retained, churned) * 1.1)
        st.pyplot(fig)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        colors_pie = ['#32CD32', '#FF6347']
        explode = (0, 0.05)
        ax2.pie([retained, churned], labels=['Retained', 'Churned'], autopct='%1.1f%%', colors=colors_pie, explode=explode, startangle=90, textprops={'fontsize': 12})
        ax2.set_title('Churn Distribution', fontsize=14, fontweight='bold')
        st.pyplot(fig2)
    
    st.markdown("---")
    
    st.subheader("📋 Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

with tab2:
    st.header("Exploratory Data Analysis")
    
    st.subheader("Churn by Contract Type")
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(8, 5))
        contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean())
        colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(contract_churn.index, contract_churn.values, color=colors_bar)
        ax.set_title('Churn Rate by Contract Type', fontsize=13, fontweight='bold')
        ax.set_ylabel('Churn Rate')
        ax.set_ylim(0, 0.5)
        for bar, val in zip(bars, contract_churn.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.1%}', ha='center', fontsize=11)
        st.pyplot(fig)
    
    with col2:
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        df.groupby('Contract')['Churn'].value_counts().unstack().plot(kind='bar', ax=ax2, color=['#32CD32', '#FF6347'])
        ax2.set_title('Customer Count by Contract & Churn', fontsize=13, fontweight='bold')
        ax2.set_xlabel('Contract Type')
        ax2.set_ylabel('Count')
        ax2.legend(title='Churn')
        plt.xticks(rotation=0)
        st.pyplot(fig2)
    
    st.subheader("Churn by Internet Service")
    col1, col2 = st.columns(2)
    
    with col1:
        fig3, ax3 = plt.subplots(figsize=(8, 5))
        internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean())
        colors_int = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax3.bar(internet_churn.index, internet_churn.values, color=colors_int)
        ax3.set_title('Churn Rate by Internet Service', fontsize=13, fontweight='bold')
        ax3.set_ylabel('Churn Rate')
        ax3.set_ylim(0, 0.5)
        for bar, val in zip(bars, internet_churn.values):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.1%}', ha='center', fontsize=11)
        st.pyplot(fig3)
    
    with col2:
        fig4, ax4 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Churn', y='tenure', data=df, palette=['#32CD32', '#FF6347'])
        ax4.set_title('Tenure Distribution by Churn Status', fontsize=13, fontweight='bold')
        ax4.set_xlabel('Churn')
        ax4.set_ylabel('Tenure (Months)')
        st.pyplot(fig4)
    
    st.subheader("Churn by Demographics")
    col1, col2 = st.columns(2)
    
    with col1:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x='gender', hue='Churn', palette=['#32CD32', '#FF6347'])
        ax5.set_title('Churn by Gender', fontsize=13, fontweight='bold')
        ax5.set_xlabel('Gender')
        ax5.set_ylabel('Count')
        st.pyplot(fig5)
    
    with col2:
        fig6, ax6 = plt.subplots(figsize=(8, 5))
        senior_churn = df.groupby('SeniorCitizen')['Churn'].apply(lambda x: (x == 'Yes').mean())
        labels = ['Non-Senior', 'Senior']
        ax6.bar(labels, senior_churn.values, color=['#4ECDC4', '#FF6B6B'])
        ax6.set_title('Churn Rate by Senior Status', fontsize=13, fontweight='bold')
        ax6.set_ylabel('Churn Rate')
        ax6.set_ylim(0, 0.5)
        for i, val in enumerate(senior_churn.values):
            ax6.text(i, val + 0.01, f'{val:.1%}', ha='center', fontsize=11)
        st.pyplot(fig6)
    
    st.subheader("Financial Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig7, ax7 = plt.subplots(figsize=(8, 5))
        sns.boxplot(x='Churn', y='MonthlyCharges', data=df, palette=['#32CD32', '#FF6347'])
        ax7.set_title('Monthly Charges by Churn Status', fontsize=13, fontweight='bold')
        ax7.set_xlabel('Churn')
        ax7.set_ylabel('Monthly Charges ($)')
        st.pyplot(fig7)
    
    with col2:
        fig8, ax8 = plt.subplots(figsize=(8, 5))
        payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean())
        ax8.barh(payment_churn.index, payment_churn.values, color='#FF6B6B')
        ax8.set_title('Churn Rate by Payment Method', fontsize=13, fontweight='bold')
        ax8.set_xlabel('Churn Rate')
        ax8.set_xlim(0, 0.5)
        for i, val in enumerate(payment_churn.values):
            ax8.text(val + 0.01, i, f'{val:.1%}', va='center', fontsize=10)
        st.pyplot(fig8)

with tab3:
    st.header("Machine Learning Models")
    
    st.subheader("Model Comparison")
    
    model_data = {
        'Model': ['Logistic Regression', 'XGBoost'],
        'Accuracy': [f"{models['lr']['accuracy']:.2%}", f"{models['xgb']['accuracy']:.2%}"],
        'Precision': [f"{models['lr']['precision']:.2%}", f"{models['xgb']['precision']:.2%}"],
        'Recall': [f"{models['lr']['recall']:.2%}", f"{models['xgb']['recall']:.2%}"],
        'F1-Score': [f"{models['lr']['f1']:.2%}", f"{models['xgb']['f1']:.2%}"],
        'ROC AUC': [f"{models['lr']['roc']:.2%}", f"{models['xgb']['roc']:.2%}"]
    }
    
    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Logistic Regression - Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 5))
        cm_lr = confusion_matrix(models['lr']['y_test'], models['lr']['preds'])
        sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        ax.set_title('Confusion Matrix')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)
    
    with col2:
        st.subheader("XGBoost - Confusion Matrix")
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        cm_xgb = confusion_matrix(models['xgb']['y_test'], models['xgb']['preds'])
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax2, xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
        ax2.set_title('Confusion Matrix')
        ax2.set_ylabel('Actual')
        ax2.set_xlabel('Predicted')
        st.pyplot(fig2)
    
    st.subheader("Feature Importance (XGBoost)")
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    feature_importance = pd.DataFrame({
        'feature': models['xgb']['X'].columns,
        'importance': models['xgb']['model'].feature_importances_
    }).sort_values('importance', ascending=True).tail(15)
    
    ax3.barh(feature_importance['feature'], feature_importance['importance'], color='#3498db')
    ax3.set_title('Top 15 Features - XGBoost', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Importance')
    st.pyplot(fig3)

with tab4:
    st.header("Business Insights & Recommendations")
    
    st.subheader("🎯 Key Findings")
    
    findings = [
        ("📊 Overall Churn Rate", f"{churn_rate:.1f}% of customers have churned ({churned:,} out of {total_customers:,})"),
        ("📋 Contract Impact", "Month-to-month contracts have the highest churn rate (~42%)"),
        ("⏱️ Tenure Effect", "Customers with tenure < 12 months are significantly more likely to churn"),
        ("🌐 Internet Service", "Fiber optic customers show the highest churn rate among internet types"),
        ("💰 Monthly Charges", "Customers with higher monthly charges tend to have higher churn"),
        ("📱 Payment Method", "Electronic check users have the highest churn rate")
    ]
    
    for title, desc in findings:
        st.info(f"**{title}**  \n\n{desc}")
    
    st.markdown("---")
    
    st.subheader("💡 Recommendations")
    
    recommendations = [
        ("1️⃣ Promote Long-term Contracts", "Offer incentives (discounts, bonus services) to encourage month-to-month customers to switch to annual/two-year contracts"),
        ("2️⃣ Early Retention Programs", "Implement special onboarding and engagement programs for customers in their first 12 months"),
        ("3️⃣ Improve Fiber Optic Service", "Investigate and address service quality issues for fiber optic customers"),
        ("4️⃣ Review Pricing Strategy", "Analyze pricing for high-value customers and consider retention offers"),
        ("5️⃣ Encourage Auto-Payment", "Promote automatic payment methods to reduce churn from electronic check users"),
        ("6️⃣ Targeted Offers", "Create personalized offers based on customer service usage and tenure")
    ]
    
    for title, desc in recommendations:
        st.success(f"**{title}**  \n\n{desc}")
    
    st.markdown("---")
    
    st.subheader("📈 Model Performance Summary")
    
    best_model = "Logistic Regression" if models['lr']['accuracy'] > models['xgb']['accuracy'] else "XGBoost"
    best_accuracy = max(models['lr']['accuracy'], models['xgb']['accuracy'])
    
    st.write(f"""
    - **Best Performing Model:** {best_model}
    - **Accuracy:** {best_accuracy:.2%}
    - **ROC AUC Score:** {max(models['lr']['roc'], models['xgb']['roc']):.2%}
    
    The model can predict customer churn with approximately **{best_accuracy:.0%} accuracy**, helping the business identify at-risk customers proactively.
    """)

st.markdown("---")
st.markdown("<div style='text-align: center; color: gray;'>📊 Telco Customer Churn Analysis | Created by Harshita Sharma</div>", unsafe_allow_html=True)