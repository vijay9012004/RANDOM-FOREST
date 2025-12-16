import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# Define the file paths
MODEL_FILE = "random_forest_model.pkl"
DATA_FILE = "Iris (1).csv"

# --- 1. Load Data and Model (Cached for Efficiency) ---
@st.cache_resource
def load_resources():
    """Loads the data and the pre-trained model."""
    
    # 1. Load the original data to determine min/max ranges for sliders
    if not os.path.exists(DATA_FILE):
        st.error(f"Error: Required data file '{DATA_FILE}' not found. Please upload the Iris data.")
        return None, None
        
    df = pd.read_csv(DATA_FILE)
    # Exclude 'Id' and the target variable 'Species' for feature min/max calculation
    X_df = df.drop(['Id', 'Species'], axis=1, errors='ignore')

    # 2. Load the saved Random Forest Model
    if not os.path.exists(MODEL_FILE):
        st.error(f"Error: Required model file '{MODEL_FILE}' not found. Please ensure it was created.")
        return None, None
        
    try:
        with open(MODEL_FILE, 'rb') as file:
            loaded_model = pickle.load(file)
        
        return loaded_model, X_df
        
    
    except Exception as e:
        st.error(f"Error loading the model from '{MODEL_FILE}'. Check if the model was saved with compatible versions of scikit-learn. Details: {e}")
        return None, None

# Load the model and feature frame
rf_model, X_df = load_resources()

# --- 2. Streamlit Application Interface ---
st.title('🌸 Iris Species Prediction App')
st.markdown("""
This application uses your pre-trained **Random Forest Classifier** (`random_forest_model.pkl`) 
to predict the species of an Iris flower based on its measurements.
""")

if rf_model is not None and X_df is not None:
    
    # --- 3. Sidebar for User Input ---
    st.sidebar.header('Input Parameters (cm)')

    def user_input_features():
        # Define sliders based on min/max of the loaded dataset
        sepal_length = st.sidebar.slider(
            'Sepal Length (SepalLengthCm)', 
            float(X_df['SepalLengthCm'].min()), 
            float(X_df['SepalLengthCm'].max()), 
            float(X_df['SepalLengthCm'].mean())
        )
        sepal_width = st.sidebar.slider(
            'Sepal Width (SepalWidthCm)', 
            float(X_df['SepalWidthCm'].min()), 
            float(X_df['SepalWidthCm'].max()), 
            float(X_df['SepalWidthCm'].mean())
        )
        petal_length = st.sidebar.slider(
            'Petal Length (PetalLengthCm)', 
            float(X_df['PetalLengthCm'].min()), 
            float(X_df['PetalLengthCm'].max()), 
            float(X_df['PetalLengthCm'].mean())
        )
        petal_width = st.sidebar.slider(
            'Petal Width (PetalWidthCm)', 
            float(X_df['PetalWidthCm'].min()), 
            float(X_df['PetalWidthCm'].max()), 
            float(X_df['PetalWidthCm'].mean())
        )
        
        # Create a DataFrame from the inputs
        data = {
            'SepalLengthCm': sepal_length,
            'SepalWidthCm': sepal_width,
            'PetalLengthCm': petal_length,
            'PetalWidthCm': petal_width
        }
        features = pd.DataFrame(data, index=[0])
        return features

    # Get user input
    input_df = user_input_features()

    st.subheader('User Input Parameters')
    st.dataframe(input_df, hide_index=True)

    # --- 4. Prediction ---
    prediction = rf_model.predict(input_df)
    prediction_proba = rf_model.predict_proba(input_df)
    
    # --- 5. Display Result ---
    st.subheader('Prediction Result')
    
    class_name = prediction[0]
    
    # Simple color formatting based on the predicted class
    if class_name == 'Iris-setosa':
        color = 'green'
    elif class_name == 'Iris-versicolor':
        color = 'orange'
    else: # Iris-virginica
        color = 'red'

    st.markdown(f"The predicted Iris Species is: <span style='font-size: 32px; color: {color}; font-weight: bold;'>{class_name}</span>", unsafe_allow_html=True)
    
    # Optional: Display prediction probabilities
    st.markdown("---")
    st.subheader('Prediction Probability')
    
    # Organize probabilities for better display
    proba_df = pd.DataFrame(
        prediction_proba.T, 
        index=rf_model.classes_, 
        columns=['Probability']
    ).sort_values(by='Probability', ascending=False)
    
    st.dataframe(proba_df, width=250)

# --- How to Run This App ---
# 1. Save the code above as a file named `app.py`.
# 2. Ensure your `random_forest_model.pkl` and `Iris (1).csv` files are in the same directory.
# 3. Open your terminal or command prompt in that directory.
# 4. Run the command: `streamlit run app.py`
