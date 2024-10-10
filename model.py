# model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

def train_model():
    # Load the dataset
    data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/electricity.csv")
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    
    # Define features (X) and target (y)
    X = data.drop('ActualWindProduction', axis=1)  # Adjust target column as needed
    y = data['ActualWindProduction']
    
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and fit the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, 'electricity_model.pkl')

if __name__ == "__main__":
    train_model()
