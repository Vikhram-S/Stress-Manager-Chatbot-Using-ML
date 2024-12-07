"""
Welcome to the Mental Stress Manager Web App!

For Users:
This app is your personal AI companion for managing stress and mental wellbeing. Here's what you can do:

1. Chat Interface:
   - Have natural conversations about your stress and concerns
   - Get personalized responses and support
   - Track your stress levels over time

2. Smart Analysis:
   - Upload your health data through CSV file
   - Get AI-powered stress level predictions
   - Receive personalized recommendations

3. Features:
   - User-friendly chat interface
   - Real-time stress assessment
   - Beautiful, calming design
   - Secure and private interactions

4. How to Use:
   - Upload your health dataset when prompted
   - Type your questions or concerns in the chat
   - Get instant responses and guidance
   - Track your progress over time

Created by Team Z Data Knights
For support, reach us through the social links below
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# Set page configuration and styling
st.set_page_config(
    page_title="Mental Stress Manager", 
    page_icon="🧠",
    layout="wide"
)

# Custom CSS styling with modern UI colors
st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stButton>button {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        margin: 5px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
        border: 2px solid #4ECDC4;
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #4ECDC4;
        background-color: rgba(255,255,255,0.1);
        color: white;
    }
    .chat-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
    }
    .chat-message {
        padding: 15px;
        border-radius: 15px;
        margin: 10px 0;
        display: flex;
        align-items: flex-start;
    }
    .user-message {
        background: linear-gradient(45deg, #FF6B6B22, #FF6B6B44);
        margin-left: 50px;
    }
    .bot-message {
        background: linear-gradient(45deg, #4ECDC422, #4ECDC444);
        margin-right: 50px;
    }
    .avatar {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        margin-right: 10px;
    }
    h1, h2, h3 {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    p {
        color: #FFFFFF;
        font-size: 16px;
        line-height: 1.6;
    }
    .result-card {
        background: linear-gradient(45deg, #2C3E50, #3498DB);
        padding: 25px;
        border-radius: 20px;
        margin: 20px 0;
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    .footer {
        background: linear-gradient(45deg, #2C3E50, #3498DB);
        padding: 30px;
        border-radius: 20px;
        margin-top: 50px;
    }
    .footer-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        max-width: 1200px;
        margin: auto;
    }
    .team-section {
        flex: 1;
        min-width: 300px;
        padding: 20px;
    }
    .social-links {
        display: flex;
        gap: 20px;
    }
    .social-link {
        color: white;
        text-decoration: none;
        padding: 10px 20px;
        border-radius: 30px;
        background: rgba(255,255,255,0.1);
        transition: all 0.3s ease;
    }
    .social-link:hover {
        background: rgba(255,255,255,0.2);
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Upload dataset
st.title("Mental Stress Manager")
uploaded_file = st.file_uploader("Upload the Sleep Health and Lifestyle Dataset (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.error("Please upload the required dataset to proceed.")
    st.stop()

# Data preprocessing
df_cleaned = df.drop_duplicates()
if 'Person ID' in df_cleaned.columns:
    df_cleaned = df_cleaned.drop(columns=['Person ID'])

# Convert categorical variables
categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']
df_cleaned[categorical_columns] = df_cleaned[categorical_columns].astype('category')

# Prepare model data
df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
X = df_encoded.drop(columns=['Stress Level'])
y = df_encoded['Stress Level']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)

def predict_stress(user_data):
    """
    Predict stress level based on user input data.
    Uses Random Forest model trained on sleep health and lifestyle data.
    Returns a stress level prediction between 1-10.
    """
    user_df = pd.DataFrame([user_data])
    user_encoded = pd.get_dummies(user_df, drop_first=True)
    user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)
    prediction = rf_model.predict(user_encoded)
    return prediction[0]

def main():
    st.markdown("<h1>Mental Stress Manager</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Your AI-powered stress management assistant</p>", unsafe_allow_html=True)

    # Chat interface
    st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class='chat-message user-message'>
                    <img src='https://api.dicebear.com/6.x/avataaars/svg?seed=user' class='avatar'>
                    <div>{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class='chat-message bot-message'>
                    <img src='https://api.dicebear.com/6.x/bottts/svg?seed=bot' class='avatar'>
                    <div>{message['content']}</div>
                </div>
            """, unsafe_allow_html=True)

    # User input
    user_input = st.text_input("Ask me about your stress levels...", key="user_input")
    
    if st.button("Send"):
        if user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Process user input and generate response
            # This is a simplified example - you would need to implement proper NLP here
            response = "I understand you're concerned about stress. Let's assess your stress levels with some questions."
            
            # Add bot response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            st.rerun()

    # Footer
    st.markdown("""
        <div class='footer'>
            <div class='footer-content'>
                <div class='team-section'>
                    <h3>Development Team</h3>
                    <p>Lead Developer: Vikhram S</p>
                    <p>Co-Developers: Ragul S, Roshan R, Nithesh Kumar B</p>
                </div>
                <div class='social-links'>
                    <a href='https://www.linkedin.com/in/vikhram-s' class='social-link'>📧 LinkedIn</a>
                    <a href='mailto:vikhrams@saveetha.ac.in' class='social-link'>💼 Email</a>
                    <a href='https://github.com/Vikhram-S' class='social-link'>🌟 GitHub</a>
                </div>
            </div>
            <p style='text-align: center; margin-top: 20px;'>Made with ❤️ by Team Z Data Knights</p>
            <p style='text-align: center; margin-top: 10px;'>© 2024 Mental Stress Manager Chatbot by Z Data Knights. All rights reserved.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
