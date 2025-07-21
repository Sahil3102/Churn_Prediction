import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
import plotly.express as px
import plotly.graph_objects as go
# from Dashboard import load_css 


# Set page configuration
st.set_page_config(
    page_title="Customer Churn Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# load_css()
# Custom CSS
def local_css():
    st.markdown("""
    <style>
        /* Main container styling */
        .main {
            background-color: #f8f9fa;
            padding: 2rem;
        }
        
        /* Header styling */
        .main-header {
            color: #1E3A8A;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1.5rem;
            text-align: center;
            padding-bottom: 1rem;
            border-bottom: 2px solid #E5E7EB;
        }
        
        /* Card styling */
        .card {
            background-color: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            padding: 1.5rem;
            margin-bottom: 2rem;
        }
        
        /* Section headers */
        .section-header {
            color: #1E3A8A;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #E5E7EB;
        }
        
        /* Metrics styling */
        .metric-container {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .metric-card {
            background-color: #EFF6FF;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            flex: 1;
            min-width: 150px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #1E40AF;
        }
        
        .metric-label {
            font-size: 0.9rem;
            color: #6B7280;
        }
        
        /* Prediction result styling */
        .prediction-yes {
            background-color: #FEE2E2;
            color: #B91C1C;
            border: 1px solid #EF4444;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .prediction-no {
            background-color: #D1FAE5;
            color: #065F46;
            border: 1px solid #10B981;
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #2563EB;
            color: white;
            font-weight: 600;
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            width: 100%;
        }
        
        .stButton>button:hover {
            background-color: #1D4ED8;
        }
        
        /* DataFrames */
        .dataframe-container {
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2rem;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 3rem;
            white-space: nowrap;
            font-size: 1rem;
            font-weight: 500;
            color: #4B5563;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #EFF6FF;
            color: #1E40AF;
            font-weight: 600;
        }
        
        /* File uploader */
        .upload-container {
            border: 2px dashed #D1D5DB;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Select box styling */
        div[data-baseweb="select"] {
            margin-bottom: 1rem;
        }
        
        /* Model selection styling */
        .model-selection {
            background-color: #F3F4F6;
            border-radius: 8px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        
        .model-card {
            background-color: white;
            border: 2px solid #E5E7EB;
            border-radius: 8px;
            padding: 1rem;
            margin-bottom: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .model-card:hover, .model-card.selected {
            border-color: #2563EB;
            box-shadow: 0 4px 6px rgba(37, 99, 235, 0.1);
        }
        
        .model-card.selected {
            background-color: #EFF6FF;
        }
        
        .model-title {
            font-weight: 600;
            color: #1F2937;
            margin-bottom: 0.5rem;
        }
        
        .model-description {
            color: #6B7280;
            font-size: 0.9rem;
        }
    </style>
    """, unsafe_allow_html=True)

def load_data():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    st.markdown('<h3 style="color: #4B5563;">Upload Your Dataset</h3>', unsafe_allow_html=True)
    st.markdown('<p style="color: #6B7280;">Upload a CSV file containing customer data with a "Churn" column</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["csv"])
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

def create_dynamic_bins(df, column, num_bins=5):
    min_val = df[column].min()
    max_val = df[column].max()
    bins = np.linspace(min_val, max_val, num_bins + 1)
    labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(num_bins)]
    return pd.cut(df[column], bins=bins, labels=labels, include_lowest=True)

def preprocess_data(df):
    df = df.copy()
    # Save original dataframe for analysis
    original_df = df.copy()
    
    # Handle CustomerID column if it exists
    if 'CustomerID' in df.columns:
        df.drop(columns=['CustomerID'], inplace=True)
    
    df.fillna(0, inplace=True)  # Handle missing values
    
    # Convert TotalCharges to numeric
    if 'TotalCharges' in df.columns:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df.fillna({'TotalCharges': df['TotalCharges'].median()}, inplace=True)
    
    # Dynamically create bins for numeric columns
    if 'Tenure' in df.columns:
        df['TenureGroup'] = create_dynamic_bins(df, 'Tenure')
    if 'MonthlyCharges' in df.columns:
        df['MonthlyChargesGroup'] = create_dynamic_bins(df, 'MonthlyCharges')
    if 'TotalCharges' in df.columns:
        df['TotalChargesGroup'] = create_dynamic_bins(df, 'TotalCharges')
    
    # Drop original numeric columns that have been binned
    numeric_cols_to_drop = []
    if 'TenureGroup' in df.columns:
        numeric_cols_to_drop.append('Tenure')
    if 'MonthlyChargesGroup' in df.columns:
        numeric_cols_to_drop.append('MonthlyCharges')
    if 'TotalChargesGroup' in df.columns:
        numeric_cols_to_drop.append('TotalCharges')
    
    if numeric_cols_to_drop:
        df.drop(columns=numeric_cols_to_drop, inplace=True)
    
    # Store original dataframe before encoding
    processed_original_df = df.copy()
    
    # Encode categorical variables
    label_encoders = {}
    for col in df.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))  # Convert categories to numbers
        label_encoders[col] = le
    
    return df, processed_original_df, label_encoders, original_df

def train_model(df, model_type):
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Model Type: {model_type}")
    # Initialize model based on selected type
    if model_type == "random_forest":
        model = RandomForestClassifier(n_estimators=200, 
                                    max_depth=10,
                                    min_samples_split=5,
                                    random_state=42)

    elif model_type == "logistic_regression":
        model = LogisticRegression(C=0.1, 
                                 penalty='l2',
                                 solver='liblinear',
                                 random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    elif model_type == "svm":
        model = SVC(C=1.0, 
                  kernel='rbf',
                  gamma='scale',
                  probability=True,
                  random_state=42)
        # Scale features for SVM
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
    else:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred) * 100
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Get classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Get model-specific metrics
    metric_data = {}
    
    # Add ROC curve data
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    metric_data['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'auc': roc_auc}
    
    # Add precision-recall curve data
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    metric_data['pr_curve'] = {'precision': precision, 'recall': recall}
    
    # Get feature importance based on model type
    feature_importance = None
    if model_type == "random_forest":
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
    elif model_type == "logistic_regression":
        # For logistic regression, use coefficients as importance
        importance = np.abs(model.coef_[0])
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
    elif model_type == "svm":
        # SVM doesn't provide feature importance directly
        # We'll use a simpler approach - train a logistic regression on the same data
        # and use its coefficients as a proxy for feature importance
        lr = LogisticRegression(max_iter=1000)
        lr.fit(X_train, y_train)
        importance = np.abs(lr.coef_[0])
        feature_importance = pd.DataFrame({
            'Feature': X.columns,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)
    
    return model, acc, cm, feature_importance, X_test, y_test, report, metric_data, X_train, y_train, X.columns

def analyze_data(df, target_col='Churn'):
    """Perform data analysis and generate insights"""
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Data Analysis</h2>', unsafe_allow_html=True)
    
    # Data Overview with metrics in cards
    st.markdown('<h3 class="section-header">Data Overview</h3>', unsafe_allow_html=True)
    
    # Create metrics display
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{df.shape[0]}</div>
            <div class="metric-label">Total Records</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df.shape[1]}</div>
            <div class="metric-label">Total Features</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df.select_dtypes(include=['object']).shape[1]}</div>
            <div class="metric-label">Categorical Features</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{df.select_dtypes(include=['int64', 'float64']).shape[1]}</div>
            <div class="metric-label">Numerical Features</div>
        </div>
    ''', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Check for target column
    if target_col in df.columns:
        # Churn Distribution
        churn_dist = df[target_col].value_counts(normalize=True) * 100
        
        fig = px.pie(values=churn_dist.values, 
                    names=churn_dist.index, 
                    title='Customer Churn Distribution (%)',
                    color_discrete_sequence=px.colors.sequential.Blues,
                    hole=0.4)
        
        fig.update_layout(
            font=dict(size=14),
            legend=dict(orientation="h", y=-0.1),
            margin=dict(t=50, b=20, l=20, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Analyze categorical features
    st.markdown('<h3 class="section-header">Categorical Features Analysis</h3>', unsafe_allow_html=True)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    # Convert categorical_cols to a list
    categorical_cols = list(categorical_cols)
    
    if target_col in categorical_cols:
        categorical_cols = [col for col in categorical_cols if col != target_col]
    
    if len(categorical_cols) > 0:  # Check if the list is not empty
        selected_cat_feature = st.selectbox("Select Categorical Feature", categorical_cols)
        
        if target_col in df.columns:
            # Create a grouped bar chart showing churn rate by selected category
            cat_churn = df.groupby([selected_cat_feature, target_col]).size().unstack()
            
            if cat_churn.shape[1] == 2:  # Binary churn (Yes/No or 1/0)
                # Calculate churn rate
                if isinstance(cat_churn.columns, pd.Index):
                    churn_col = cat_churn.columns[1]  # Assuming "Yes" or 1 is the second column
                    no_churn_col = cat_churn.columns[0]
                else:
                    # Handle if columns aren't an index
                    churn_col = list(cat_churn.columns)[1]
                    no_churn_col = list(cat_churn.columns)[0]
                
                cat_churn['Churn Rate'] = cat_churn[churn_col] / (cat_churn[churn_col] + cat_churn[no_churn_col]) * 100
                
                fig = px.bar(cat_churn.reset_index(), 
                            x=selected_cat_feature, 
                            y='Churn Rate',
                            title=f'Churn Rate by {selected_cat_feature}',
                            color_discrete_sequence=['#2563EB'])
                
                fig.update_layout(
                    xaxis_title=selected_cat_feature,
                    yaxis_title="Churn Rate (%)",
                    font=dict(size=12),
                    margin=dict(t=50, b=50, l=50, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Stacked bar chart
                fig = px.bar(df, 
                            x=selected_cat_feature, 
                            color=target_col,
                            title=f'Distribution of {selected_cat_feature} by Churn Status',
                            barmode='stack',
                            color_discrete_map={
                                'Yes': '#EF4444', 
                                'No': '#10B981',
                                1: '#EF4444',
                                0: '#10B981'
                            })
                
                fig.update_layout(
                    xaxis_title=selected_cat_feature,
                    yaxis_title="Count",
                    legend_title="Churn Status",
                    font=dict(size=12),
                    margin=dict(t=50, b=50, l=50, r=20)
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Simple value counts for the selected feature
            value_counts = df[selected_cat_feature].value_counts()
            fig = px.bar(x=value_counts.index, 
                         y=value_counts.values,
                         title=f'Distribution of {selected_cat_feature}',
                         color_discrete_sequence=['#2563EB'])
            fig.update_layout(
                xaxis_title=selected_cat_feature,
                yaxis_title="Count",
                font=dict(size=12),
                margin=dict(t=50, b=50, l=50, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Analyze numerical features
    st.markdown('<h3 class="section-header">Numerical Features Analysis</h3>', unsafe_allow_html=True)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Convert numeric_cols to a list
    numeric_cols = list(numeric_cols)
    
    if len(numeric_cols) > 0:  # Check if the list is not empty
        selected_num_feature = st.selectbox("Select Numerical Feature", numeric_cols)
        
        # Histogram
        fig = px.histogram(df, 
                          x=selected_num_feature,
                          color=target_col if target_col in df.columns else None,
                          marginal="box",
                          title=f'Distribution of {selected_num_feature}',
                          color_discrete_map={
                              'Yes': '#EF4444', 
                              'No': '#10B981',
                              1: '#EF4444',
                              0: '#10B981'
                          })
        
        fig.update_layout(
            xaxis_title=selected_num_feature,
            yaxis_title="Count",
            legend_title="Churn Status" if target_col in df.columns else None,
            font=dict(size=12),
            margin=dict(t=50, b=50, l=50, r=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # If we have a target column, show the box plot
        if target_col in df.columns:
            fig = px.box(df, 
                        x=target_col, 
                        y=selected_num_feature,
                        title=f'{selected_num_feature} by {target_col}',
                        color=target_col,
                        color_discrete_map={
                            'Yes': '#EF4444', 
                            'No': '#10B981',
                            1: '#EF4444',
                            0: '#10B981'
                        })
            
            fig.update_layout(
                xaxis_title="Churn Status",
                yaxis_title=selected_num_feature,
                font=dict(size=12),
                margin=dict(t=50, b=50, l=50, r=20)
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_model_insights(cm, feature_importance, X_test, y_test, report, model_type, metric_data):
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-header">Model Insights</h2>', unsafe_allow_html=True)
    
    # Create two columns for layout
    col1, col2 = st.columns(2)
    
    # Classification Report as metrics
    col1.markdown('<h3 class="section-header">Classification Metrics</h3>', unsafe_allow_html=True)
    
    # Get key metrics
    precision = report['weighted avg']['precision'] * 100
    recall = report['weighted avg']['recall'] * 100
    f1 = report['weighted avg']['f1-score'] * 100
    
    # Create metrics display
    col1.markdown('<div class="metric-container">', unsafe_allow_html=True)
    col1.markdown(f'''
        <div class="metric-card">
            <div class="metric-value">{precision:.1f}%</div>
            <div class="metric-label">Precision</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{recall:.1f}%</div>
            <div class="metric-label">Recall</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{f1:.1f}%</div>
            <div class="metric-label">F1 Score</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{metric_data['roc_curve']['auc']:.3f}</div>
            <div class="metric-label">AUC-ROC</div>
        </div>
    ''', unsafe_allow_html=True)
    col1.markdown('</div>', unsafe_allow_html=True)
    
    # Confusion Matrix
    col1.markdown('<h3 class="section-header">Confusion Matrix</h3>', unsafe_allow_html=True)
    fig = px.imshow(cm, 
                   text_auto=True,
                   labels=dict(x="Predicted", y="Actual", color="Count"),
                   x=['No Churn', 'Churn'],
                   y=['No Churn', 'Churn'],
                   color_continuous_scale='Blues')
    
    fig.update_layout(
        width=400,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    col1.plotly_chart(fig)
    
    # Feature Importance
    if feature_importance is not None:
        col2.markdown('<h3 class="section-header">Feature Importance</h3>', unsafe_allow_html=True)
        fig = px.bar(feature_importance.head(10), 
                    x='Importance', 
                    y='Feature',
                    orientation='h',
                    title='Top 10 Features by Importance',
                    color='Importance',
                    color_continuous_scale='Blues')
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="",
            yaxis={'categoryorder':'total ascending'},
            font=dict(size=12),
            margin=dict(t=50, b=50, l=150, r=20)
        )
        col2.plotly_chart(fig)
    
    # ROC curve
    col2.markdown('<h3 class="section-header">ROC Curve</h3>', unsafe_allow_html=True)
    fig = px.line(
        x=metric_data['roc_curve']['fpr'], 
        y=metric_data['roc_curve']['tpr'],
        labels={'x': 'False Positive Rate', 'y': 'True Positive Rate'},
        title=f'ROC Curve (AUC = {metric_data["roc_curve"]["auc"]:.3f})'
    )
    
    # Add diagonal line
    fig.add_shape(
        type='line', 
        line=dict(dash='dash', color='gray'),
        x0=0, x1=1, y0=0, y1=1
    )
    
    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        font=dict(size=12),
        margin=dict(t=50, b=50, l=50, r=20)
    )
    col2.plotly_chart(fig)
    
    # Full classification report
    st.markdown('<h3 class="section-header">Detailed Classification Report</h3>', unsafe_allow_html=True)
    report_df = pd.DataFrame(report).T
    
    # Style the dataframe
    st.markdown('<div class="dataframe-container">', unsafe_allow_html=True)
    st.dataframe(report_df.style.background_gradient(cmap='Blues', subset=['precision', 'recall', 'f1-score']), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model-specific additional insights
    st.markdown('<h3 class="section-header">Model-Specific Insights</h3>', unsafe_allow_html=True)
    
    # Create two columns for additional insights
    add_col1, add_col2 = st.columns(2)
    
    # Precision-Recall Curve
    add_col1.markdown('<h4>Precision-Recall Curve</h4>', unsafe_allow_html=True)
    fig = px.line(
        x=metric_data['pr_curve']['recall'], 
        y=metric_data['pr_curve']['precision'],
        labels={'x': 'Recall', 'y': 'Precision'},
        title='Precision-Recall Curve'
    )
    
    fig.update_layout(
        xaxis_title="Recall",
        yaxis_title="Precision",
        font=dict(size=12),
        margin=dict(t=50, b=50, l=50, r=20)
    )
    add_col1.plotly_chart(fig)
    
    # Model-specific additional visualization
    if model_type == "random_forest":
        add_col2.markdown('<h4>Random Forest Insights</h4>', unsafe_allow_html=True)
        add_col2.markdown("""
        <p style="color: #4B5563;">
            <strong>Key Characteristics:</strong>
            <ul>
                <li>Ensemble learning method using multiple decision trees</li>
                <li>Handles non-linear relationships well</li>
                <li>Robust to outliers and missing values</li>
                <li>Less prone to overfitting compared to single decision trees</li>
                <li>Can provide reliable feature importance scores</li>
            </ul>
        </p>
        """, unsafe_allow_html=True)
    elif model_type == "logistic_regression":
        add_col2.markdown('<h4>Logistic Regression Insights</h4>', unsafe_allow_html=True)
        add_col2.markdown("""
        <p style="color: #4B5563;">
            <strong>Key Characteristics:</strong>
            <ul>
                <li>Linear model for classification problems</li>
                <li>Outputs probabilities between 0 and 1</li>
                <li>Highly interpretable through coefficient values</li>
                <li>Works best with linearly separable data</li>
                <li>Less complex and faster to train than ensemble methods</li>
            </ul>
        </p>
        """, unsafe_allow_html=True)
    elif model_type == "svm":
        add_col2.markdown('<h4>Support Vector Machine Insights</h4>', unsafe_allow_html=True)
        add_col2.markdown("""
        <p style="color: #4B5563;">
            <strong>Key Characteristics:</strong>
            <ul>
                <li>Finds optimal hyperplane to separate classes</li>
                <li>Effective in high-dimensional spaces</li>
                <li>Versatile through different kernel functions</li>
                <li>Robust to overfitting in high-dimensional spaces</li>
                <li>Memory efficient as it uses only a subset of training points (support vectors)</li>
            </ul>
        </p>
        """, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Apply custom CSS
    local_css()
    
    # Main header
    st.markdown('<h1 class="main-header">Customer Churn Analysis & Prediction</h1>', unsafe_allow_html=True)
    
    # Introduction text
    st.markdown('''
        <div style="text-align: center; margin-bottom: 2rem; color: #4B5563;">
            <p>Upload your customer data to analyze churn patterns and predict which customers are at risk of churning.</p>
            <p>This tool helps you understand key drivers of customer churn and take proactive retention measures.</p>
        </div>
    ''', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    
    if df is not None:
        # Check if the dataset contains a Churn column
        if 'Churn' not in df.columns:
            st.error("The uploaded dataset must contain a 'Churn' column for analysis.")
            return
        
        # Create tabs for different sections
        tab1, tab2, tab3 = st.tabs(["Data Analysis", "Model Training", "Predict Churn"])
        
        with tab1:
            # Analyze the data
            analyze_data(df)
        
        with tab2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Model Training</h2>', unsafe_allow_html=True)
            
            # Preprocess the data
            processed_df, processed_original_df, label_encoders, original_df = preprocess_data(df)
            
            # Model selection
            st.markdown('<h3 class="section-header">Select Model Type</h3>', unsafe_allow_html=True)

            # Initialize selected_model in session state if not exists
            if 'selected_model' not in st.session_state:
                st.session_state.selected_model = "random_forest"

            # Create model selection cards
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown('<div class="model-card" id="rf-card">', unsafe_allow_html=True)
                st.markdown('<h4 class="model-title">Random Forest</h4>', unsafe_allow_html=True)
                st.markdown('<p class="model-description">Ensemble method using multiple decision trees. Good for complex relationships.</p>', unsafe_allow_html=True)
                if st.button("Select Random Forest", key="rf_btn"):
                    st.session_state.selected_model = "random_forest"
                st.markdown('</div>', unsafe_allow_html=True)

            with col2:
                st.markdown('<div class="model-card" id="lr-card">', unsafe_allow_html=True)
                st.markdown('<h4 class="model-title">Logistic Regression</h4>', unsafe_allow_html=True)
                st.markdown('<p class="model-description">Linear model for classification. Highly interpretable.</p>', unsafe_allow_html=True)
                if st.button("Select Logistic Regression", key="lr_btn"):
                    st.session_state.selected_model = "logistic_regression"
                st.markdown('</div>', unsafe_allow_html=True)

            with col3:
                st.markdown('<div class="model-card" id="svm-card">', unsafe_allow_html=True)
                st.markdown('<h4 class="model-title">Support Vector Machine</h4>', unsafe_allow_html=True)
                st.markdown('<p class="model-description">Finds optimal hyperplane to separate classes. Good for high-dimensional data.</p>', unsafe_allow_html=True)
                if st.button("Select SVM", key="svm_btn"):
                    st.session_state.selected_model = "svm"
                st.markdown('</div>', unsafe_allow_html=True)

            # Display the currently selected model
            st.markdown(f'<p>Selected Model: <strong>{st.session_state.selected_model}</strong></p>', unsafe_allow_html=True)

            # Train the model
            if st.button("Train Model", key="train_btn"):
                with st.spinner("Training model..."):
                    model, acc, cm, feature_importance, X_test, y_test, report, metric_data, X_train, y_train, feature_names = train_model(processed_df, st.session_state.selected_model)
                    
                    # Rest of your training code...
                    
                    # Show model performance
                    st.success(f"Model trained successfully!")
                    
                    # Display accuracy
                    st.markdown(f'''
                        <div class="metric-container">
                            <div class="metric-card">
                                <div class="metric-value">{acc:.1f}%</div>
                                <div class="metric-label">Test Accuracy</div>
                            </div>
                        </div>
                    ''', unsafe_allow_html=True)
                    
                    # Show model insights
                    show_model_insights(cm, feature_importance, X_test, y_test, report, st.session_state.selected_model, metric_data)
                    
                    # Store model and related data in session state
                    st.session_state['model'] = model
                    st.session_state['label_encoders'] = label_encoders
                    st.session_state['feature_names'] = feature_names
                    st.session_state['processed_original_df'] = processed_original_df
                    st.session_state['original_df'] = original_df
            st.markdown('</div>', unsafe_allow_html=True)
        
        with tab3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">Predict Customer Churn</h2>', unsafe_allow_html=True)
            
            # Check if model is trained
            if 'model' not in st.session_state:
                st.warning("Please train a model first in the 'Model Training' tab.")
            else:
                # Get model and encoders from session state
                model = st.session_state['model']
                label_encoders = st.session_state['label_encoders']
                feature_names = st.session_state['feature_names']
                processed_original_df = st.session_state['processed_original_df']
                original_df = st.session_state['original_df']
                
                # Create form for prediction
                st.markdown('<h3 class="section-header">Enter Customer Details</h3>', unsafe_allow_html=True)
                
                # Get sample customer data for reference
                sample_customer = processed_original_df.iloc[0].copy()
                
                # Create input form
                input_data = {}
                cols = st.columns(2)
                
                for i, feature in enumerate(feature_names):
                    if feature == 'Churn':
                        continue
                        
                    col = cols[i % 2]
                    
                    # Handle categorical features
                    if feature in label_encoders:
                        categories = list(label_encoders[feature].classes_)
                        input_data[feature] = col.selectbox(
                            f"{feature}",
                            options=categories,
                            index=0,
                            key=f"input_{feature}"
                        )
                    else:
                        # For numerical features
                        min_val = float(processed_original_df[feature].min())
                        max_val = float(processed_original_df[feature].max())
                        default_val = float(sample_customer[feature]) if feature in sample_customer else (min_val + max_val) / 2
                        
                        input_data[feature] = col.number_input(
                            f"{feature}",
                            min_value=min_val,
                            max_value=max_val,
                            value=default_val,
                            key=f"input_{feature}"
                        )
                
                # Make prediction
                if st.button("Predict Churn", key="predict_btn"):
                    # Prepare input data for prediction
                    prediction_data = {}
                    for feature in feature_names:
                        if feature == 'Churn':
                            continue
                            
                        if feature in label_encoders:
                            # Encode categorical features
                            le = label_encoders[feature]
                            prediction_data[feature] = le.transform([input_data[feature]])[0]
                        else:
                            prediction_data[feature] = input_data[feature]
                    
                    # Convert to DataFrame
                    prediction_df = pd.DataFrame([prediction_data])
                    
                    # Ensure columns are in correct order - use only features that exist in the dataframe
                    prediction_df = prediction_df[[col for col in feature_names if col != 'Churn']]
                    
                    # Make prediction
                    proba = model.predict_proba(prediction_df)[0]
                    prediction = model.predict(prediction_df)[0]
                    
                    
                    # Display prediction result
                    churn_prob = proba[1] * 100
                    no_churn_prob = proba[0] * 100
                    
                    if prediction == 1:
                        st.markdown(f'''
                            <div class="prediction-yes">
                                <h3>Prediction: CHURN RISK</h3>
                                <p>This customer has a <strong>{churn_prob:.1f}%</strong> probability of churning.</p>
                            </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown(f'''
                            <div class="prediction-no">
                                <h3>Prediction: NO CHURN RISK</h3>
                                <p>This customer has a <strong>{no_churn_prob:.1f}%</strong> probability of staying.</p>
                            </div>
                        ''', unsafe_allow_html=True)
                    
                    # Show probability breakdown
                    fig = go.Figure(go.Bar(
                        x=[no_churn_prob, churn_prob],
                        y=['No Churn', 'Churn'],
                        orientation='h',
                        marker_color=['#10B981', '#EF4444']
                    ))
                    
                    fig.update_layout(
                        title="Churn Probability Breakdown",
                        xaxis_title="Probability (%)",
                        yaxis_title="Prediction",
                        height=300,
                        margin=dict(l=50, r=50, t=50, b=50)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show feature importance for this prediction (if available)
                    if hasattr(model, 'feature_importances_'):
                        st.markdown('<h3 class="section-header">Key Factors in This Prediction</h3>', unsafe_allow_html=True)
                        
                        # Get SHAP values if possible
                        try:
                            import shap
                            explainer = shap.TreeExplainer(model)
                            shap_values = explainer.shap_values(prediction_df)
                            
                            # Plot SHAP values
                            fig, ax = plt.subplots()
                            shap.summary_plot(shap_values[1], prediction_df, plot_type="bar", show=False)
                            st.pyplot(fig)
                            plt.close()
                        except:
                            # Fallback to regular feature importance
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(importance_df.head(10), 
                                        x='Importance', 
                                        y='Feature',
                                        orientation='h',
                                        title='Top Features Affecting Prediction',
                                        color='Importance',
                                        color_continuous_scale='Blues')
                            
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()