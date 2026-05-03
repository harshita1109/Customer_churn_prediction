import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import xgboost as xgb
import os
import warnings
warnings.filterwarnings('ignore')

THEME = st.sidebar.radio("🎨 Theme", ["Dark", "Light"], index=0)

if THEME == "Dark":
    BG_COLOR = "#0f0c29"
    CARD_BG = "rgba(255,255,255,0.05)"
    TEXT_COLOR = "white"
    GRID_COLOR = "rgba(255,255,255,0.1)"
    METRIC_COLORS = {
        'blue': ['#667eea', '#764ba2'],
        'green': ['#11998e', '#38ef7d'],
        'red': ['#eb3349', '#f45c43'],
        'orange': ['#f093fb', '#f5576c']
    }
    PLOT_BG = "rgba(0,0,0,0)"
    CHURN_COLORS = ['#38ef7d', '#f45c43']
else:
    BG_COLOR = "#f8f9fa"
    CARD_BG = "rgba(0,0,0,0.05)"
    TEXT_COLOR = "#1a1a2e"
    GRID_COLOR = "rgba(0,0,0,0.1)"
    METRIC_COLORS = {
        'blue': ['#4a90d9', '#67b3e8'],
        'green': ['#2ecc71', '#58d68d'],
        'red': ['#e74c3c', '#ec7063'],
        'orange': ['#f39c12', '#f7dc6f']
    }
    PLOT_BG = "white"
    CHURN_COLORS = ['#2ecc71', '#e74c3c']

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background: {BG_COLOR};
        min-height: 100vh;
    }}
    
    .main-content {{
        background: {CARD_BG};
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 20px;
        margin: 10px;
    }}
    
    .metric-card {{
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
        transition: transform 0.3s ease;
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
    }}
    
    .metric-card.blue {{
        background: linear-gradient(135deg, {METRIC_COLORS['blue'][0]} 0%, {METRIC_COLORS['blue'][1]} 100%);
    }}
    
    .metric-card.green {{
        background: linear-gradient(135deg, {METRIC_COLORS['green'][0]} 0%, {METRIC_COLORS['green'][1]} 100%);
    }}
    
    .metric-card.red {{
        background: linear-gradient(135deg, {METRIC_COLORS['red'][0]} 0%, {METRIC_COLORS['red'][1]} 100%);
    }}
    
    .metric-card.orange {{
        background: linear-gradient(135deg, {METRIC_COLORS['orange'][0]} 0%, {METRIC_COLORS['orange'][1]} 100%);
    }}
    
    .metric-value {{
        font-size: 2.5em;
        font-weight: 700;
        color: white;
    }}
    
    .metric-label {{
        font-size: 0.9em;
        color: rgba(255,255,255,0.9);
        text-transform: uppercase;
        letter-spacing: 1px;
    }}
    
    .section-header {{
        color: {TEXT_COLOR};
        font-size: 1.8em;
        font-weight: 600;
        margin-bottom: 20px;
        padding-bottom: 10px;
        border-bottom: 2px solid {'rgba(255,255,255,0.2)' if THEME == 'Dark' else 'rgba(0,0,0,0.1)'};
    }}
    
    .insight-card {{
        background: {CARD_BG};
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }}
    
    .insight-card.success {{ border-left-color: {METRIC_COLORS['green'][0]}; }}
    .insight-card.warning {{ border-left-color: {METRIC_COLORS['red'][0]}; }}
    .insight-card.info {{ border-left-color: {METRIC_COLORS['blue'][0]}; }}
    
    .stMarkdown p, .stMarkdown li {{
        color: {TEXT_COLOR} !important;
    }}
    
    h1, h2, h3, h4, h5, h6 {{
        color: {TEXT_COLOR} !important;
    }}
    
    .stTab [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    
    .stDataFrame {{
        background: {CARD_BG} !important;
        border-radius: 10px !important;
    }}
    
    div[data-testid="stSidebar"] {{
        background: {'linear-gradient(180deg, #1a1a2e 0%, #16213e 100%)' if THEME == 'Dark' else 'linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%)'} !important;
    }}
    
    .sidebar-section {{
        background: {CARD_BG};
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
    }}
    
    .stButton > button {{
        background: linear-gradient(135deg, {METRIC_COLORS['blue'][0]}, {METRIC_COLORS['blue'][1]}) !important;
        color: white !important;
        border-radius: 10px !important;
        border: none !important;
    }}
    
    .stButton > button:hover {{
        transform: scale(1.02);
    }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.set_page_config(
    page_title="Telco Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    
    df_processed = df.drop('customerID', axis=1).copy()
    le = LabelEncoder()
    df_processed['Churn'] = le.fit_transform(df_processed['Churn'])
    categorical_cols = df_processed.select_dtypes(include='object').columns.tolist()
    df_processed = pd.get_dummies(df_processed, columns=categorical_cols, drop_first=True)
    
    return df, df_processed

def train_models(df_processed):
    X = df_processed.drop('Churn', axis=1)
    y = df_processed['Churn']
    
    scaler = StandardScaler()
    X[['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices']] = scaler.fit_transform(
        X[['tenure', 'MonthlyCharges', 'TotalCharges', 'TotalServices']])
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)
    
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, 
                                   random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train, y_train)
    
    return {
        'lr': {'model': lr_model, 'accuracy': accuracy_score(y_test, lr_model.predict(X_test)),
               'precision': precision_score(y_test, lr_model.predict(X_test)),
               'recall': recall_score(y_test, lr_model.predict(X_test)),
               'f1': f1_score(y_test, lr_model.predict(X_test)),
               'roc': roc_auc_score(y_test, lr_model.predict_proba(X_test)[:, 1])},
        'xgb': {'model': xgb_model, 'accuracy': accuracy_score(y_test, xgb_model.predict(X_test)),
                'precision': precision_score(y_test, xgb_model.predict(X_test)),
                'recall': recall_score(y_test, xgb_model.predict(X_test)),
                'f1': f1_score(y_test, xgb_model.predict(X_test)),
                'roc': roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])},
        'X_test': X_test, 'y_test': y_test, 'X': X
    }

def create_metric_card(value, label, card_type="blue"):
    return f"""
    <div class="metric-card {card_type}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def get_chart_layout(title):
    return dict(
        title=dict(text=title, font=dict(size=16, color=TEXT_COLOR)),
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(gridcolor=GRID_COLOR),
        yaxis=dict(gridcolor=GRID_COLOR)
    )

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/chart-growth.png", width=80)
    st.markdown(f"<h2 style='color: {TEXT_COLOR};'>📊 Churn Analytics</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown(f"### 🎛️ Filters", unsafe_allow_html=True)
    contract_filter = st.multiselect(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        default=["Month-to-month", "One year", "Two year"]
    )
    
    internet_filter = st.multiselect(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"],
        default=["DSL", "Fiber optic", "No"]
    )
    
    st.markdown("---")
    st.markdown(f"### ℹ️ Project Info", unsafe_allow_html=True)
    st.info(f"""
    **Telco Customer Churn Analysis**
    
    - Dataset: 7,043 customers
    - Models: Logistic Regression & XGBoost
    - Best Accuracy: ~80%
    """)
    
    st.markdown(f"### 🔗 Quick Links", unsafe_allow_html=True)
    st.markdown(f"- [📊 GitHub Repo](https://github.com/harshita1109/Customer_churn_prediction)")

df, df_processed = load_and_prepare_data()
models = train_models(df_processed)

total_customers = len(df)
churned = (df['Churn'] == 'Yes').sum()
retained = (df['Churn'] == 'No').sum()
churn_rate = (churned / total_customers) * 100

st.markdown("<div class='main-content'>", unsafe_allow_html=True)

st.markdown(f"<h1 style='text-align: center; font-size: 3em; color: {TEXT_COLOR};'>📊 Telco Customer Churn Analysis</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='text-align: center; color: {'rgba(255,255,255,0.7)' if THEME == 'Dark' else 'rgba(0,0,0,0.6)'}; font-size: 1.2em;'>Predictive Analytics for Customer Retention | Powered by Machine Learning</p>", unsafe_allow_html=True)
st.markdown("---")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["🏠 Dashboard", "📈 EDA Analysis", "🤖 Models", "💡 Insights", "🔮 Predict"])

with tab1:
    st.markdown(f"<div class='section-header'>📊 Executive Summary</div>", unsafe_allow_html=True)
    
    cols = st.columns(4)
    with cols[0]:
        st.markdown(create_metric_card(f"{total_customers:,}", "Total Customers", "blue"), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(create_metric_card(f"{churned:,}", "Churned", "red"), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(create_metric_card(f"{retained:,}", "Retained", "green"), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(create_metric_card(f"{churn_rate:.1f}%", "Churn Rate", "orange"), unsafe_allow_html=True)
    
    st.markdown("###")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[go.Pie(
            labels=['Retained', 'Churned'],
            values=[retained, churned],
            marker=dict(colors=CHURN_COLORS),
            textinfo='percent+label',
            textfont=dict(size=14, color='white'),
            hole=0.6,
            pull=[0, 0.1]
        )])
        fig.update_layout(**get_chart_layout('Customer Distribution'))
        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = go.Figure(go.Bar(
            x=['Retained', 'Churned'],
            y=[retained, churned],
            marker=dict(color=CHURN_COLORS, line=dict(color='white', width=2)),
            text=[f'{retained:,}', f'{churned:,}'],
            textposition='outside',
            textfont=dict(size=16, color=TEXT_COLOR)
        ))
        fig2.update_layout(**get_chart_layout('Churn Count'))
        fig2.update_layout(showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("###")
    st.markdown(f"<div class='section-header'>🔥 Key Risk Factors</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    contract_churn = df.groupby('Contract')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    with col1:
        fig = go.Figure(go.Bar(
            x=contract_churn.index,
            y=contract_churn.values,
            marker=dict(color=['#f45c43', '#f093fb', '#38ef7d']),
            text=[f'{v:.1f}%' for v in contract_churn.values],
            textposition='outside'
        ))
        fig.update_layout(**get_chart_layout('Churn by Contract'))
        fig.update_layout(yaxis=dict(title='Churn Rate %', gridcolor=GRID_COLOR))
        st.plotly_chart(fig, use_container_width=True)
    
    internet_churn = df.groupby('InternetService')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    with col2:
        fig2 = go.Figure(go.Bar(
            x=internet_churn.index,
            y=internet_churn.values,
            marker=dict(color=['#4facfe', '#f45c43', '#38ef7d']),
            text=[f'{v:.1f}%' for v in internet_churn.values],
            textposition='outside'
        ))
        fig2.update_layout(**get_chart_layout('Churn by Internet'))
        fig2.update_layout(yaxis=dict(title='Churn Rate %', gridcolor=GRID_COLOR))
        st.plotly_chart(fig2, use_container_width=True)
    
    payment_churn = df.groupby('PaymentMethod')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
    with col3:
        fig3 = go.Figure(go.Bar(
            y=payment_churn.index,
            x=payment_churn.values,
            marker=dict(color='#f45c43'),
            text=[f'{v:.1f}%' for v in payment_churn.values],
            textposition='outside',
            orientation='h'
        ))
        fig3.update_layout(**get_chart_layout('Churn by Payment'))
        fig3.update_layout(xaxis=dict(title='Churn Rate %', gridcolor=GRID_COLOR))
        st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.markdown(f"<div class='section-header'>📈 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    
    st.subheader("👥 Demographics Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        gender_churn = df.groupby(['gender', 'Churn']).size().unstack()
        fig = go.Figure(data=[
            go.Bar(name='No Churn', x=gender_churn.index, y=gender_churn['No'], marker_color=CHURN_COLORS[0]),
            go.Bar(name='Churned', x=gender_churn.index, y=gender_churn['Yes'], marker_color=CHURN_COLORS[1])
        ])
        fig.update_layout(barmode='stack', **get_chart_layout('Churn by Gender'))
        fig.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        senior_data = df.groupby(['SeniorCitizen', 'Churn']).size().unstack()
        senior_data.index = ['Non-Senior', 'Senior']
        fig2 = go.Figure(data=[
            go.Bar(name='No Churn', x=senior_data.index, y=senior_data['No'], marker_color=CHURN_COLORS[0]),
            go.Bar(name='Churned', x=senior_data.index, y=senior_data['Yes'], marker_color=CHURN_COLORS[1])
        ])
        fig2.update_layout(barmode='stack', **get_chart_layout('Churn by Senior Status'))
        fig2.update_layout(legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("⏱️ Tenure Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[df['Churn']=='No']['tenure'], name='Retained', marker_color=CHURN_COLORS[0]))
        fig.add_trace(go.Box(y=df[df['Churn']=='Yes']['tenure'], name='Churned', marker_color=CHURN_COLORS[1]))
        fig.update_layout(**get_chart_layout('Tenure Distribution by Churn'))
        fig.update_layout(yaxis=dict(title='Tenure (Months)'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        tenure_bins = pd.cut(df['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12', '12-24', '24-48', '48-72'])
        tenure_churn = df.groupby(tenure_bins)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        fig2 = go.Figure(go.Scatter(
            x=tenure_churn.index.astype(str),
            y=tenure_churn.values,
            mode='lines+markers',
            marker=dict(size=12, color='#667eea', line=dict(color='white', width=2)),
            line=dict(color='#667eea', width=3),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.3)'
        ))
        fig2.update_layout(**get_chart_layout('Churn Rate by Tenure'))
        fig2.update_layout(yaxis=dict(title='Churn Rate %', gridcolor=GRID_COLOR))
        fig2.update_layout(xaxis=dict(title='Tenure (Months)'))
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("💰 Financial Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Box(y=df[df['Churn']=='No']['MonthlyCharges'], name='Retained', marker_color=CHURN_COLORS[0]))
        fig.add_trace(go.Box(y=df[df['Churn']=='Yes']['MonthlyCharges'], name='Churned', marker_color=CHURN_COLORS[1]))
        fig.update_layout(**get_chart_layout('Monthly Charges by Churn'))
        fig.update_layout(yaxis=dict(title='Monthly Charges ($)'))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig2 = px.scatter(df, x='tenure', y='MonthlyCharges', color='Churn',
                          color_discrete_map={'No': CHURN_COLORS[0], 'Yes': CHURN_COLORS[1]})
        fig2.update_layout(**get_chart_layout('Tenure vs Monthly Charges'))
        st.plotly_chart(fig2, use_container_width=True)

with tab3:
    st.markdown(f"<div class='section-header'>🤖 Machine Learning Models</div>", unsafe_allow_html=True)
    
    st.subheader("📊 Model Performance Comparison")
    
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
    lr_scores = [models['lr']['accuracy'], models['lr']['precision'], models['lr']['recall'], 
                 models['lr']['f1'], models['lr']['roc']]
    xgb_scores = [models['xgb']['accuracy'], models['xgb']['precision'], models['xgb']['recall'],
                  models['xgb']['f1'], models['xgb']['roc']]
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=lr_scores + [lr_scores[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='Logistic Regression',
        line_color='#667eea'
    ))
    fig.add_trace(go.Scatterpolar(
        r=xgb_scores + [xgb_scores[0]],
        theta=metrics + [metrics[0]],
        fill='toself',
        name='XGBoost',
        line_color='#38ef7d'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR)),
        paper_bgcolor=PLOT_BG,
        font=dict(color=TEXT_COLOR),
        showlegend=True,
        legend=dict(orientation="h", y=1.1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        metrics_df = pd.DataFrame({
            'Metric': metrics,
            'Logistic Reg': [f'{s:.2%}' for s in lr_scores],
            'XGBoost': [f'{s:.2%}' for s in xgb_scores]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    with col2:
        best_model = 'Logistic Regression' if models['lr']['accuracy'] >= models['xgb']['accuracy'] else 'XGBoost'
        best_score = max(models['lr']['accuracy'], models['xgb']['accuracy'])
        
        fig2 = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = best_score * 100,
            title = {"text": f"Best Model: {best_model}", "font": {"size": 20, "color": TEXT_COLOR}},
            gauge = {
                'axis': {'range': [0, 100], 'tickcolor': TEXT_COLOR},
                'bar': {'color': "#38ef7d"},
                'bgcolor': "rgba(0,0,0,0.3)",
                'borderwidth': 2,
                'bordercolor': "white",
            },
            number = {'suffix': "%", 'font': {"size": 40, "color": TEXT_COLOR}}
        ))
        fig2.update_layout(paper_bgcolor=PLOT_BG, font=dict(color=TEXT_COLOR))
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("🎯 Feature Importance")
    feature_imp = pd.DataFrame({
        'feature': models['X'].columns,
        'importance': models['xgb']['model'].feature_importances_
    }).sort_values('importance', ascending=True).tail(15)
    
    fig = go.Figure(go.Bar(
        x=feature_imp['importance'],
        y=feature_imp['feature'],
        orientation='h',
        marker=dict(color=px.colors.sequential.Viridis)
    ))
    fig.update_layout(**get_chart_layout('Top 15 Features (XGBoost)'))
    fig.update_layout(xaxis=dict(title='Importance', gridcolor=GRID_COLOR))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.markdown(f"<div class='section-header'>💡 Business Insights & Recommendations</div>", unsafe_allow_html=True)
    
    insights = [
        ("📊 Overall Churn Rate", f"{churn_rate:.1f}% ({churned:,} out of {total_customers:,} customers)", "info"),
        ("📋 Contract Risk", "Month-to-month contracts have ~42% churn rate - highest risk segment", "warning"),
        ("⏱️ Tenure Impact", "Customers with tenure <12 months are 3x more likely to churn", "warning"),
        ("🌐 Internet Service", "Fiber optic users show highest churn (41%) vs DSL (19%)", "warning"),
        ("💰 Payment Method", "Electronic check users churn at 45% - highest among payment types", "warning"),
        ("🎯 Model Accuracy", "80% accuracy in predicting churn - enables proactive retention", "success")
    ]
    
    for title, desc, card_type in insights:
        st.markdown(f"""
        <div class="insight-card {card_type}">
            <h3 style="color: {TEXT_COLOR};">{title}</h3>
            <p style="color: {'rgba(255,255,255,0.8)' if THEME == 'Dark' else 'rgba(0,0,0,0.7)'}">{desc}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown(f"<div class='section-header'>🎯 Strategic Recommendations</div>", unsafe_allow_html=True)
    
    recommendations = [
        ("1️⃣ Convert Month-to-Month to Annual", "Offer 20% discount for annual contracts - target 3000+ high-risk customers", "success"),
        ("2️⃣ Early Retention Program", "Create welcome kits + check-ins for first 6 months - critical retention window", "success"),
        ("3️⃣ Fiber Optic Quality Audit", "Investigate service issues - 41% churn indicates potential quality problems", "warning"),
        ("4️⃣ Auto-Payment Migration", "Offer $5/month discount for automatic payment methods - reduces churn 15%", "info"),
        ("5️⃣ High-Value Customer VIP", "Personal account manager for customers paying >$70/month with tenure <12", "info"),
        ("6️⃣ Proactive Churn Alerts", "Deploy ML model for real-time churn probability scoring - intervene before departure", "success")
    ]
    
    for title, desc, rec_type in recommendations:
        color = '#38ef7d' if rec_type == 'success' else '#f45c43' if rec_type == 'warning' else '#4facfe'
        st.markdown(f"""
        <div style="background: {CARD_BG}; border-radius: 10px; padding: 15px; margin: 10px 0; border-left: 4px solid {color};">
            <h4 style="color: {TEXT_COLOR}; margin: 0;">{title}</h4>
            <p style="color: {'rgba(255,255,255,0.8)' if THEME == 'Dark' else 'rgba(0,0,0,0.7)'}; margin: 5px 0 0 0;">{desc}</p>
        </div>
        """, unsafe_allow_html=True)

with tab5:
    st.markdown(f"<div class='section-header'>🔮 Customer Churn Predictor</div>", unsafe_allow_html=True)
    st.markdown("Enter customer details to predict churn probability")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.number_input("Monthly Charges ($)", 0, 200, 70)
        total_charges = st.number_input("Total Charges ($)", 0, 10000, 500)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    
    with col2:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        senior = st.selectbox("Senior Citizen", ["No", "Yes"])
        partner = st.selectbox("Partner", ["No", "Yes"])
    
    if st.button("🔮 Predict Churn Probability", use_container_width=True):
        with st.spinner('Running prediction model...'):
            input_data = {
                'tenure': tenure,
                'MonthlyCharges': monthly_charges,
                'TotalCharges': total_charges,
                'Contract_One year': 1 if contract == "One year" else 0,
                'Contract_Two year': 1 if contract == "Two year" else 0,
                'InternetService_Fiber optic': 1 if internet == "Fiber optic" else 0,
                'InternetService_No': 1 if internet == "No" else 0,
                'SeniorCitizen': 1 if senior == "Yes" else 0,
                'Partner': 1 if partner == "Yes" else 0
            }
            
            feature_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract_One year', 
                           'Contract_Two year', 'InternetService_Fiber optic', 'InternetService_No',
                           'SeniorCitizen', 'Partner']
            
            for col in models['X'].columns:
                if col not in feature_cols:
                    input_data[col] = 0
            
            X_input = pd.DataFrame([input_data])[models['X'].columns]
            
            prob = models['xgb']['model'].predict_proba(X_input)[0][1]
            
            color = '#38ef7d' if prob < 0.3 else '#f093fb' if prob < 0.7 else '#f45c43'
            risk = 'Low' if prob < 0.3 else 'Medium' if prob < 0.7 else 'High'
            
            st.markdown(f"""
            <div style="text-align: center; padding: 30px; background: {CARD_BG}; border-radius: 20px;">
                <h2 style="color: {TEXT_COLOR}; margin-bottom: 10px;">Churn Probability</h2>
                <div style="font-size: 4em; font-weight: bold; color: {color};">{prob:.1%}</div>
                <div style="font-size: 1.5em; color: {color}; margin-top: 10px;">{risk} Risk</div>
            </div>
            """, unsafe_allow_html=True)
            
            if prob > 0.5:
                st.warning("⚠️ This customer has high churn risk! Consider retention offers.")
            elif prob > 0.3:
                st.info("ℹ️ This customer has medium churn risk. Monitor closely.")
            else:
                st.success("✅ This customer has low churn risk. Good retention!")

st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {'rgba(255,255,255,0.6)' if THEME == 'Dark' else 'rgba(0,0,0,0.5)'}; padding: 20px;">
    <p>📊 Telco Customer Churn Analysis | Created by Harshita Sharma</p>
    <p>Powered by Python, Streamlit & Machine Learning</p>
</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)