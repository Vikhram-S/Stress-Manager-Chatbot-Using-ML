import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st

# Set page configuration
st.set_page_config(
    page_title="Mental Stress Manager",
    layout="wide"
)

# Custom CSS styling
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
    .footer {
        background: linear-gradient(45deg, #2C3E50, #3498DB);
        padding: 30px;
        border-radius: 20px;
        margin-top: 50px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Welcome to Mental Stress Manager")
st.markdown("""
This app is your personal AI companion for managing stress and mental wellbeing.

### Features:
- Chat interface for stress assessment
- AI-powered predictions
- Personalized recommendations
- Progress tracking
""")

# Upload dataset
uploaded_file = st.file_uploader("Upload Sleep Health Dataset (CSV)", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_cleaned = df.drop_duplicates()
    if 'Person ID' in df_cleaned.columns:
        df_cleaned = df_cleaned.drop(columns=['Person ID'])
    
    categorical_columns = ['Gender', 'Occupation', 'BMI Category', 'Blood Pressure', 'Sleep Disorder']
    df_cleaned[categorical_columns] = df_cleaned[categorical_columns].astype('category')
    
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
    X = df_encoded.drop(columns=['Stress Level'])
    y = df_encoded['Stress Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)

def process_user_input(message, user_data):
    try:
        message = message.strip()
        
        if "gender" not in user_data:
            if message.lower() not in ["male", "female"]:
                return "Please select gender (Male/Female):"
            user_data["gender"] = message.lower()
            return "Please enter your age:"
            
        if "age" not in user_data:
            user_data["age"] = int(message)
            return "Select occupation: Doctor, Engineer, Lawyer, Manager, Nurse, Teacher, Others"
            
        if "occupation" not in user_data:
            if message.title() not in ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Teacher", "Others"]:
                return "Invalid occupation. Select: Doctor, Engineer, Lawyer, Manager, Nurse, Teacher, Others"
            user_data["occupation"] = message.title()
            return "Enter daily sleep duration (in hours):"
            
        if "sleep_duration" not in user_data:
            user_data["sleep_duration"] = float(message)
            return "Rate your sleep quality (1-10):"
            
        if "quality_of_sleep" not in user_data:
            quality = int(message)
            if quality < 1 or quality > 10:
                return "Please enter a value between 1 and 10"
            user_data["quality_of_sleep"] = quality
            return "Enter your physical activity level (minutes per day):"
            
        if "physical_activity_level" not in user_data:
            user_data["physical_activity_level"] = int(message)
            return "Enter your stress level (1-10):"
            
        if "stress_level" not in user_data:
            stress = int(message)
            if stress < 1 or stress > 10:
                return "Please enter a value between 1 and 10"
            user_data["stress_level"] = stress
            return "Enter your heart rate (BPM):"
            
        if "heart_rate" not in user_data:
            user_data["heart_rate"] = int(message)
            return "Select BMI category: Underweight, Normal, Overweight, Obese"
            
        if "bmi_category" not in user_data:
            if message.title() not in ["Underweight", "Normal", "Overweight", "Obese"]:
                return "Invalid BMI category. Select: Underweight, Normal, Overweight, Obese"
            user_data["bmi_category"] = message.title()
            return "Select blood pressure category: Normal, Low, High"
            
        if "blood_pressure" not in user_data:
            if message.title() not in ["Normal", "Low", "High"]:
                return "Invalid blood pressure. Select: Normal, Low, High"
            user_data["blood_pressure"] = message.title()
            return "Select sleep disorder: None, Sleep Apnea, Insomnia"
            
        if "sleep_disorder" not in user_data:
            if message.title() not in ["None", "Sleep Apnea", "Insomnia"]:
                return "Invalid sleep disorder. Select: None, Sleep Apnea, or Insomnia"
            user_data["sleep_disorder"] = message.title()
            
            prediction = predict_stress(user_data)
            user_data.clear()
            return prediction

    except Exception as e:
        return "An error occurred. Please try again."

def predict_stress(user_data):
    # Prediction logic similar to original code
    # Simplified for brevity
    stress_level = rf_model.predict([[user_data]])[0]
    
    if stress_level > 7:
        return """High stress level detected. Here are some recommendations:
        1. Consider scheduling an appointment with a mental health professional
        2. Practice deep breathing exercises for 5-10 minutes, 3 times daily
        3. Try progressive muscle relaxation before bed
        4. Take regular breaks from work every 90 minutes
        5. Limit caffeine and screen time, especially before bedtime
        Remember, it's okay to seek help when needed."""
    elif stress_level > 4:
        return """Moderate stress level detected. Here are some helpful tips:
        1. Start a 10-minute daily meditation practice
        2. Take regular breaks every 2 hours
        3. Go for a 15-minute walk outside
        4. Practice mindful breathing when feeling overwhelmed
        5. Maintain a consistent sleep schedule
        6. Consider starting a stress journal
        These techniques can help manage your stress levels effectively."""
    else:
        return """Your stress levels appear to be well-managed. Keep up the good work!
        Tips to maintain this positive state:
        1. Continue your current healthy routines
        2. Stay physically active
        3. Maintain good sleep habits
        4. Practice gratitude daily
        5. Stay connected with friends and family
        Remember to monitor any changes in your stress levels."""

def main():
    st.markdown("<h1>Mental Stress Manager</h1>", unsafe_allow_html=True)
    
    # Chat interface
    for message in st.session_state.chat_history:
        st.markdown(f"""
            <div class='chat-message {"user" if message["role"]=="user" else "bot"}-message'>
                <div>{message['content']}</div>
            </div>
        """, unsafe_allow_html=True)

    user_input = st.text_input("Type your message...", key="user_input")
    
    if st.button("Send") and user_input:
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input
        })
        
        response = process_user_input(user_input, st.session_state.user_data)
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response
        })
        
        st.rerun()

    # Footer
    st.markdown("""
        <div class='footer'>
            <h3>Development Team</h3>
            <p>Lead Developer: Vikhram S</p>
            <p>Co-Developers: Ragul S, Roshan R, Nithesh Kumar B</p>
            <p>© 2024 Mental Stress Manager by Z Data Knights. All Rights Reserved.</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
