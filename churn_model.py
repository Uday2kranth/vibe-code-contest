"""
Customer Churn Prediction Model
A clean and simple implementation for predicting customer churn
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

class ChurnPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        
    def load_and_preprocess_data(self, filepath='Churn_Modelling.csv'):
        """Load and preprocess the churn dataset"""
        df = pd.read_csv(filepath)
        
        # Remove irrelevant columns for prediction
        columns_to_drop = ['RowNumber', 'CustomerId', 'Surname']
        df = df.drop(columns_to_drop, axis=1)
        
        # Separate features and target
        X = df.drop('Exited', axis=1)
        y = df['Exited']
        
        # Store feature columns for later use
        self.feature_columns = X.columns.tolist()
        
        # Encode categorical variables
        categorical_columns = ['Geography', 'Gender']
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            self.label_encoders[col] = le
        
        return X, y
    
    def train_model(self, X, y):
        """Train the churn prediction model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions and evaluate
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def predict_churn(self, customer_data):
        """Predict churn for a single customer"""
        # Create DataFrame with the same structure as training data
        df = pd.DataFrame([customer_data], columns=self.feature_columns)
        
        # Encode categorical variables using fitted encoders
        for col, encoder in self.label_encoders.items():
            if col in df.columns:
                df[col] = encoder.transform([customer_data[col]])
        
        # Scale the features
        df_scaled = self.scaler.transform(df)
        
        # Make prediction
        prediction = self.model.predict(df_scaled)[0]
        probability = self.model.predict_proba(df_scaled)[0]
        
        return {
            'prediction': int(prediction),
            'churn_probability': float(probability[1]),
            'retention_probability': float(probability[0])
        }
    
    def save_model(self, filepath='churn_model.pkl'):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath='churn_model.pkl'):
        """Load a previously trained model"""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            print(f"Model loaded from {filepath}")
            return True
        return False

def main():
    """Main function to train and save the model"""
    predictor = ChurnPredictor()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y = predictor.load_and_preprocess_data()
    
    # Train the model
    print("Training the model...")
    accuracy = predictor.train_model(X, y)
    
    # Save the trained model
    predictor.save_model()
    
    print(f"\nModel training completed with accuracy: {accuracy:.4f}")
    print("Model saved as 'churn_model.pkl'")

if __name__ == "__main__":
    main()