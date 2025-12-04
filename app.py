import streamlit as st
import pandas as pd
import numpy as np
import requests
import zipfile
import io
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# --- Page Config ---
st.set_page_config(page_title="AI Maintenance Dashboard", layout="wide")

# --- Header ---
st.title("🏭 AI-Powered Predictive Maintenance Dashboard")
st.markdown("""
**Team Exceptional (Group 1)** | ITAI 2272 Capstone  
*Real-time monitoring of industrial turbofan engines using AI.*
""")

# --- Data Loading & Processing (Cached for Speed) ---
@st.cache_data
def load_and_prep_data():
    # Download Data (Mirroring the Notebook process)
    url = "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"
    response = requests.get(url)
    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall()
    
    # Load Data
    col_names = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f'sensor{i}' for i in range(1, 22)]
    train_df = pd.read_csv('train_FD001.txt', sep=' ', header=None, names=col_names, index_col=False)
    train_df = train_df.dropna(axis=1)
    
    # Feature Engineering (RUL & Label)
    train_df['rul'] = train_df.groupby('unit')['cycle'].transform(lambda x: x.max() - x)
    train_df['failure_label'] = (train_df['rul'] <= 30).astype(int)
    
    # Simple Feature Selection (Based on Correlation Heatmap analysis)
    # We select sensors with high variance and correlation to failure
    features = ['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 'sensor11', 'sensor12', 'sensor13', 'sensor15', 'sensor17', 'sensor20', 'sensor21']
    
    return train_df, features

@st.cache_resource
def train_model(df, features):
    # Train a Random Forest for the Demo (Fast & Interpretable)
    X = df[features]
    y = df['failure_label']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    model.fit(X_scaled, y)
    
    return model, scaler

# --- Main App Logic ---
try:
    with st.spinner('Initializing System & Training AI Models...'):
        df, features = load_and_prep_data()
        model, scaler = train_model(df, features)
    
    st.success("System Online: Models Trained Successfully")

    # --- Sidebar: Engine Selection ---
    st.sidebar.header("Control Panel")
    selected_unit = st.sidebar.selectbox("Select Engine Unit ID", df['unit'].unique())
    
    # Filter data for selected unit
    unit_data = df[df['unit'] == selected_unit]
    
    # Slider to simulate time passing
    max_cycle = unit_data['cycle'].max()
    current_cycle = st.sidebar.slider("Current Operational Cycle", 1, max_cycle, max_cycle)
    
    # Get current sensor readings
    current_data = unit_data[unit_data['cycle'] == current_cycle]
    
    # --- Predictions ---
    if not current_data.empty:
        # Prepare input
        input_data = current_data[features]
        input_scaled = scaler.transform(input_data)
        
        # Predict
        fail_prob = model.predict_proba(input_scaled)[0][1]
        current_rul = current_data['rul'].values[0]
        
        # --- Dashboard Layout ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="Current Cycle", value=f"{current_cycle}")
        
        with col2:
            st.metric(label="True RUL (Historical)", value=f"{current_rul} cycles")
            
        with col3:
            # Dynamic Color for Failure Probability
            color = "normal"
            if fail_prob > 0.7: color = "nverse" 
            st.metric(label="AI Failure Probability", value=f"{fail_prob:.1%}", delta_color=color)

        # --- Alert System ---
        if fail_prob > 0.5:
            st.error(f"⚠️ CRITICAL ALERT: High probability of failure detected for Engine {selected_unit}!")
        else:
            st.success(f"✅ STATUS NORMAL: Engine {selected_unit} is operating within safety parameters.")

        # --- Visualizations ---
        st.subheader("Sensor Degradation Analysis")
        
        # Plot specific sensor degradation
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(data=unit_data, x='cycle', y='sensor11', ax=ax, label='Sensor 11 (Pressure)')
        sns.lineplot(data=unit_data, x='cycle', y='sensor4', ax=ax, label='Sensor 4 (Temperature)')
        
        # Add a vertical line for current time
        plt.axvline(x=current_cycle, color='red', linestyle='--', label='Current Time')
        plt.title(f"Sensor Telemetry over Time for Unit {selected_unit}")
        plt.ylabel("Normalized Sensor Value")
        plt.legend()
        st.pyplot(fig)

        st.info("Note: As cycle count increases (x-axis), sensor readings drift, indicating wear and tear.")

except Exception as e:
    st.error(f"An error occurred: {e}")
