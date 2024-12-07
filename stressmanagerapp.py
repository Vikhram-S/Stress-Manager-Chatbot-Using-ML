import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import traceback

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
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

st.title("Welcome to Mental Stress Manager")
st.markdown("""
This app is your personal AI companion for managing stress and mental wellbeing.

### Features:
- Chat interface for stress assessment
- AI-powered predictions
- Personalized recommendations
- Progress tracking
""")

# Define unique occupations
unique_occupations = ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Teacher", "Others"]

def process_user_input(message, user_data):
    try:
        message = message.strip()
        
        if "gender" not in user_data:
            user_data["gender"] = message.strip().lower()
            return "Great! Thank you for sharing. Could you please tell me your age?"
        
        if "age" not in user_data:
            user_data["age"] = int(message.strip())
            return f"Thank you! Now, Please select your occupation from the following options:\n{', '.join(unique_occupations)}\nIf your occupation is not listed, type 'Others'."
        
        if "occupation" not in user_data:
            if message.strip().title() not in unique_occupations:
                return f"Please select a valid occupation from the list or type 'Others'.\n{', '.join(unique_occupations)}"
            user_data["occupation"] = message.strip().title()
            return "How many hours do you sleep per day?"
        
        if "sleep_duration" not in user_data:
            user_data["sleep_duration"] = float(message.strip())
            return "Thanks! Rate the quality of your sleep on a scale of 1-10."
        
        if "quality_of_sleep" not in user_data:
            quality = int(message.strip())
            if quality < 1 or quality > 10:
                return "Please enter a value between 1 and 10"
            user_data["quality_of_sleep"] = quality
            return "I appreciate your honesty! Now, Rate your physical activity level on a scale of 1-10."
        
        if "activity_level" not in user_data:
            activity = int(message.strip())
            if activity < 1 or activity > 10:
                return "Please enter a value between 1 and 10"
            user_data["activity_level"] = activity
            return "Thank you for your response! Please choose a valid BMI category (Normal, Overweight, Obese)."
        
        if "bmi_category" not in user_data:
            if message.strip().title() not in ["Normal", "Overweight", "Obese"]:
                return "Please choose a valid BMI category (Normal, Overweight, Obese)."
            user_data["bmi_category"] = message.strip().title()
            return "Thank you! and What is your systolic blood pressure?"
        
        if "systolic_bp" not in user_data:
            user_data["systolic_bp"] = float(message.strip())
            return "Thank you for your response! What is your diastolic blood pressure?"
        
        if "diastolic_bp" not in user_data:
            user_data["diastolic_bp"] = float(message.strip())
            return "Thanks! Can you please tell me your heart rate?"
        
        if "heart_rate" not in user_data:
            user_data["heart_rate"] = float(message.strip())
            return "Almost there! How many steps do you take daily?"
        
        if "daily_steps" not in user_data:
            user_data["daily_steps"] = int(message.strip())
            return "Thank you for sharing! Do you have any sleep disorders? (None, Sleep Apnea, Insomnia)"
        
        if "sleep_disorder" not in user_data:
            if message.strip().title() not in ["None", "Sleep Apnea", "Insomnia"]:
                return "I appreciate your input! Please choose a valid sleep disorder option (None, Sleep Apnea, Insomnia)."
            user_data["sleep_disorder"] = message.strip().title()

            # Calculate predicted stress level
            predicted_stress = calculate_stress_level(user_data)
            
            # Format response with predicted stress level
            response = f"""Based on your inputs, your predicted stress level is: {predicted_stress:.1f} out of 10\n\n"""
            
            if predicted_stress > 7:
                response += """High stress level detected. Here are some recommendations:
                1. Consider scheduling an appointment with a mental health professional
                2. Practice deep breathing exercises for 5-10 minutes, 3 times daily
                3. Try progressive muscle relaxation before bed
                4. Take regular breaks from work every 90 minutes
                5. Limit caffeine and screen time, especially before bedtime
                Remember, it's okay to seek help when needed."""
            elif predicted_stress > 4:
                response += """Moderate stress level detected. Here are some helpful tips:
                1. Start a 10-minute daily meditation practice
                2. Take regular breaks every 2 hours
                3. Go for a 15-minute walk outside
                4. Practice mindful breathing when feeling overwhelmed
                5. Maintain a consistent sleep schedule
                6. Consider starting a stress journal
                These techniques can help manage your stress levels effectively."""
            else:
                response += """Your stress levels appear to be well-managed. Keep up the good work!
                Tips to maintain this positive state:
                1. Continue your current healthy routines
                2. Stay physically active
                3. Maintain good sleep habits
                4. Practice gratitude daily
                5. Stay connected with friends and family
                Remember to monitor any changes in your stress levels."""
            
            user_data.clear()  # Reset for next user
            return response

    except Exception as e:
        print(traceback.format_exc())
        return "An error occurred while processing your input. Please try again."

def calculate_stress_level(user_data):
    # This is a simplified calculation - replace with your actual stress prediction logic
    base_stress = 5.0
    
    # Adjust based on sleep quality and duration
    if user_data["quality_of_sleep"] < 5:
        base_stress += 1.5
    if user_data["sleep_duration"] < 6:
        base_stress += 1.0
        
    # Adjust based on activity level
    if user_data["activity_level"] < 5:
        base_stress += 1.0
    
    # Adjust based on BMI
    if user_data["bmi_category"] in ["Overweight", "Obese"]:
        base_stress += 0.5
        
    # Adjust based on sleep disorder
    if user_data["sleep_disorder"] != "None":
        base_stress += 1.0
        
    # Ensure stress level stays within 1-10 range
    return max(1.0, min(10.0, base_stress))

def clear_chat():
    st.session_state.chat_history = []
    st.session_state.user_data = {}
    st.session_state.user_input = ""

def main():
    st.markdown("<h1>Mental Stress Manager</h1>", unsafe_allow_html=True)
    
    # Chat interface
    for message in st.session_state.chat_history:
        st.markdown(f"""
            <div class='chat-message {"user" if message["role"]=="user" else "bot"}-message'>
                <div>{message['content']}</div>
            </div>
        """, unsafe_allow_html=True)

    # Add clear chat button
    if st.button("Clear Chat"):
        clear_chat()
        st.rerun()

    # Text input with auto-clear
    user_input = st.text_input("Type your message...", key="user_input", value="")
    
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
        
        # Clear the input after sending
        st.session_state.user_input = ""
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
