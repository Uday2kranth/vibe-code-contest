"""
Customer Churn Prediction Streamlit App - Deployment Ready
A simple and clean interface for predicting customer churn
Optimized for cloud deployment with robust error handling
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Configure page
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for vibrant, polished styling
st.markdown("""
<style>
    /* Main header styling */
    .main-header {
        font-size: 3.2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%);
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 2rem;
        background: linear-gradient(135deg, #667eea08 0%, #764ba208 100%);
        border-radius: 15px;
        margin-top: 1rem;
    }
    
    /* Success prediction card */
    .prediction-success {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        border: none;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Warning prediction card */
    .prediction-warning {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        border: none;
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    /* Buttons styling */
    .stButton > button {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #6d28d9 100%);
        box-shadow: 0 4px 15px rgba(139, 92, 246, 0.4);
        transform: translateY(-1px);
    }
    
    /* Animation for cards */
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-top: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

class DeploymentChurnPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.feature_columns = [
            'CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 
            'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary'
        ]
        
    def load_model(self):
        """Load model with multiple fallback options for deployment"""
        model_paths = [
            'churn_model.pkl',
            './churn_model.pkl',
            os.path.join(os.getcwd(), 'churn_model.pkl')
        ]
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        model_data = pickle.load(f)
                    
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.label_encoders = model_data['label_encoders']
                    self.feature_columns = model_data['feature_columns']
                    st.success(f"Model loaded successfully from {path}")
                    return True
            except Exception as e:
                st.warning(f"Failed to load from {path}: {str(e)}")
                continue
        
        # If no model found, create and train a new one
        return self.create_and_train_model()
    
    def create_and_train_model(self):
        """Create and train model if not found - for deployment"""
        try:
            st.info("Model not found. Training a new model...")
            
            # Check if dataset exists
            if not os.path.exists('Churn_Modelling.csv'):
                st.error("Dataset 'Churn_Modelling.csv' not found. Please ensure the dataset is available.")
                return False
            
            # Load and train model
            df = pd.read_csv('Churn_Modelling.csv')
            
            # Remove irrelevant columns
            columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
            df = df.drop(columns_to_drop, axis=1)
            
            # Separate features and target
            X = df.drop('Exited', axis=1)
            y = df['Exited']
            
            # Initialize components
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            
            # Encode categorical variables
            categorical_columns = ['Geography', 'Gender']
            for col in categorical_columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                self.label_encoders[col] = le
            
            # Train model
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
            
            # Save the model
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns
            }
            
            with open('churn_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            st.success("New model trained and saved successfully!")
            return True
            
        except Exception as e:
            st.error(f"Failed to create model: {str(e)}")
            return False
    
    def predict_churn(self, customer_data):
        """Predict churn for a single customer"""
        try:
            # Create DataFrame
            df = pd.DataFrame([customer_data], columns=self.feature_columns)
            
            # Encode categorical variables
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    df[col] = encoder.transform([customer_data[col]])
            
            # Scale features
            df_scaled = self.scaler.transform(df)
            
            # Make prediction
            prediction = self.model.predict(df_scaled)[0]
            probability = self.model.predict_proba(df_scaled)[0]
            
            return {
                'prediction': int(prediction),
                'churn_probability': float(probability[1]),
                'retention_probability': float(probability[0])
            }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

@st.cache_resource
def load_predictor():
    """Load predictor with caching for better performance"""
    predictor = DeploymentChurnPredictor()
    if predictor.load_model():
        return predictor
    return None

def create_input_form():
    """Create input form for customer data"""
    st.sidebar.header("Customer Information")
    
    credit_score = st.sidebar.slider("Credit Score", 300, 850, 650, help="Customer's credit score")
    geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"], help="Customer's country")
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"], help="Customer's gender")
    age = st.sidebar.slider("Age", 18, 100, 40, help="Customer's age")
    tenure = st.sidebar.slider("Tenure (years)", 0, 10, 5, help="Number of years as customer")
    balance = st.sidebar.number_input("Account Balance ($)", 0.0, 300000.0, 100000.0, step=1000.0, help="Current account balance")
    num_products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], help="Number of bank products")
    has_credit_card = st.sidebar.selectbox("Has Credit Card", ["Yes", "No"], help="Does customer have a credit card")
    is_active_member = st.sidebar.selectbox("Is Active Member", ["Yes", "No"], help="Is customer an active member")
    estimated_salary = st.sidebar.number_input("Estimated Salary ($)", 0.0, 200000.0, 100000.0, step=1000.0, help="Customer's estimated salary")
    
    customer_data = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_products,
        'HasCrCard': 1 if has_credit_card == "Yes" else 0,
        'IsActiveMember': 1 if is_active_member == "Yes" else 0,
        'EstimatedSalary': estimated_salary
    }
    
    return customer_data

def create_prediction_visualization(prediction_result):
    """Create visualization for prediction results"""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = prediction_result['churn_probability'] * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Churn Probability (%)", 'font': {'size': 20, 'color': '#1e293b'}},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100], 'tickcolor': '#475569'},
            'bar': {'color': "#8b5cf6", 'thickness': 0.3},
            'steps': [
                {'range': [0, 25], 'color': "#10b981"},
                {'range': [25, 50], 'color': "#f59e0b"},
                {'range': [50, 75], 'color': "#ef4444"},
                {'range': [75, 100], 'color': "#dc2626"}
            ],
            'threshold': {
                'line': {'color': "#7c3aed", 'width': 4},
                'thickness': 0.8,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#1e293b', 'size': 14}
    )
    return fig

def display_prediction_result(prediction_result):
    """Display prediction results with styling"""
    churn_prob = prediction_result['churn_probability']
    retention_prob = prediction_result['retention_probability']
    
    if prediction_result['prediction'] == 1:
        st.markdown(f"""
        <div class="prediction-warning">
            <h3>High Churn Risk Customer</h3>
            <p>This customer has a <strong>{churn_prob:.1%}</strong> probability of churning.</p>
            <p><strong>Recommendation:</strong> Immediate retention action recommended!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="prediction-success">
            <h3>Low Churn Risk Customer</h3>
            <p>This customer has a <strong>{retention_prob:.1%}</strong> probability of staying.</p>
            <p><strong>Recommendation:</strong> Continue with standard customer care.</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">Customer Churn Prediction System</h1>', unsafe_allow_html=True)
    
    # Description
    st.markdown("""
    ### Predict Customer Churn with Machine Learning
    
    This application uses a Random Forest model to predict whether a customer is likely to churn (leave the bank).
    Simply enter the customer information in the sidebar and get an instant prediction!
    """)
    
    # Load predictor
    predictor = load_predictor()
    
    if predictor is None:
        st.error("Failed to load or create the prediction model. Please check the logs above.")
        st.stop()
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Model Information")
        st.info("""
        **Model:** Random Forest Classifier
        **Features:** 10 customer attributes
        **Target:** Churn (Yes/No)
        **Status:** Ready for predictions
        """)
        
        # Get customer input
        customer_data = create_input_form()
        
        # Prediction button
        if st.sidebar.button("Predict Churn", type="primary"):
            with st.spinner("Making prediction..."):
                prediction_result = predictor.predict_churn(customer_data)
                if prediction_result:
                    st.session_state.prediction = prediction_result
                    st.session_state.customer_data = customer_data
                    st.rerun()
    
    with col2:
        if hasattr(st.session_state, 'prediction') and st.session_state.prediction:
            st.subheader("Prediction Results")
            
            # Display metrics
            col2_1, col2_2 = st.columns(2)
            
            with col2_1:
                st.metric(
                    "Churn Probability", 
                    f"{st.session_state.prediction['churn_probability']:.1%}",
                    delta=f"{st.session_state.prediction['churn_probability'] - 0.5:.1%}"
                )
            
            with col2_2:
                st.metric(
                    "Retention Probability", 
                    f"{st.session_state.prediction['retention_probability']:.1%}",
                    delta=f"{st.session_state.prediction['retention_probability'] - 0.5:.1%}"
                )
            
            # Display prediction result
            display_prediction_result(st.session_state.prediction)
            
            # Visualization
            fig = create_prediction_visualization(st.session_state.prediction)
            st.plotly_chart(fig, use_container_width=True)
            
            # Customer data summary
            with st.expander("Customer Data Summary"):
                df_display = pd.DataFrame([st.session_state.customer_data]).T
                df_display.columns = ['Value']
                df_display['Value'] = df_display['Value'].astype(str)
                st.dataframe(df_display, use_container_width=True)
        else:
            st.subheader("Enter Customer Information")
            st.info("Please fill in the customer information in the sidebar and click 'Predict Churn' to see the results.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class='footer'>
        <p style='margin: 0; font-size: 1.1rem; font-weight: 500;'>Built with Streamlit and Scikit-learn</p>
        <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>2024 Customer Churn Prediction System</p>
        <p style='margin: 0.5rem 0 0 0; font-size: 1rem; font-weight: 600;'>AV College of Arts, Science and Commerce</p>
        <p style='margin: 0; font-size: 0.9rem; opacity: 0.9;'>Team AV</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()