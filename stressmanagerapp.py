import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import traceback

# Set page configuration and styling
st.set_page_config(
    page_title="Mental Stress Manager", 
    page_icon="🧠",
    layout="wide"
)

# Initialize user data dictionary in session state if not exists
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}

# Display welcome message and app info
st.title("Welcome to Mental Stress Manager 🧠")
st.markdown("""
This is your personal AI companion for managing stress and mental wellbeing.

### What You Can Do:
- Have natural conversations about your stress and concerns
- Get personalized responses and support 
- Track your stress levels over time
- Upload health data for AI-powered predictions
- Receive customized recommendations

### Key Features:
- User-friendly chat interface
- Real-time stress assessment
- Beautiful, calming design
- Secure and private interactions
""")

# Define unique occupations
unique_occupations = ["Doctor", "Engineer", "Lawyer", "Manager", "Nurse", "Teacher", "Others"]

def process_user_input(message, user_data):
    """
    Process user input and return appropriate response based on conversation flow
    """
    try:
        # Step through each question based on user responses
        if "gender" not in user_data:
            user_data["gender"] = message
            return "Great! Thank you for sharing. Could you please tell me your age?"
        
        if "age" not in user_data:
            user_data["age"] = int(message)
            return f"Thank you! Now, Please select your occupation from the following options:\n{', '.join(unique_occupations)}\nIf your occupation is not listed, type 'Others'."
        
        if "occupation" not in user_data:
            if message not in unique_occupations:
                return f"Please select a valid occupation from the list or type 'Others'.\n{', '.join(unique_occupations)}"
            user_data["occupation"] = message
            return "How many hours do you sleep per day?"
        
        if "sleep_duration" not in user_data:
            user_data["sleep_duration"] = float(message)
            return "Thanks! Rate the quality of your sleep on a scale of 1-10."
        
        if "quality_of_sleep" not in user_data:
            user_data["quality_of_sleep"] = int(message)
            return "I appreciate your honesty! Now, Rate your physical activity level on a scale of 1-10."
        
        if "activity_level" not in user_data:
            user_data["activity_level"] = int(message)
            return "Thank you for your response! Please choose a valid BMI category (Normal, Overweight, Obese)."
        
        if "bmi_category" not in user_data:
            if message not in ["Normal", "Overweight", "Obese"]:
                return "Please choose a valid BMI category (Normal, Overweight, Obese)."
            user_data["bmi_category"] = message
            return "Thank you! and What is your systolic blood pressure?"
        
        if "systolic_bp" not in user_data:
            user_data["systolic_bp"] = float(message)
            return "Thank you for your response! What is your diastolic blood pressure?"
        
        if "diastolic_bp" not in user_data:
            user_data["diastolic_bp"] = float(message)
            return "Thanks! Can you please tell me your heart rate?"
        
        if "heart_rate" not in user_data:
            user_data["heart_rate"] = float(message)
            return "Almost there! How many steps do you take daily?"
        
        if "daily_steps" not in user_data:
            user_data["daily_steps"] = int(message)
            return "Thank you for sharing! Do you have any sleep disorders? (None, Sleep Apnea, Insomnia)"
        
        if "sleep_disorder" not in user_data:
            if message not in ["None", "Sleep Apnea", "Insomnia"]:
                return "I appreciate your input! Please choose a valid sleep disorder option (None, Sleep Apnea, Insomnia)."
            user_data["sleep_disorder"] = message

            # All data collected, proceed to prediction
            prediction = predict_stress(
                user_data["gender"], user_data["age"], user_data["occupation"],
                user_data["sleep_duration"], user_data["quality_of_sleep"], 
                user_data["activity_level"], user_data["bmi_category"], 
                user_data["systolic_bp"], user_data["diastolic_bp"], 
                user_data["heart_rate"], user_data["daily_steps"], 
                user_data["sleep_disorder"]
            )
            user_data.clear()  # Reset for the next user
            return prediction

    except Exception as e:
        print(traceback.format_exc())
        return "An error occurred while processing your input. Please try again."

def predict_stress(gender, age, occupation, sleep_duration, quality_of_sleep, activity_level, 
                   bmi_category, systolic_bp, diastolic_bp, heart_rate, daily_steps, sleep_disorder):
    
    input_data = pd.DataFrame({
        'Age': [age],
        'Sleep Duration': [sleep_duration],
        'Quality of Sleep': [quality_of_sleep],
        'Physical Activity Level': [activity_level],
        'Heart Rate': [heart_rate],
        'Daily Steps': [daily_steps],
        'Gender_Male': [1 if gender == "Male" else 0],
        'Systolic_BP': [systolic_bp],
        'Diastolic_BP': [diastolic_bp]
    })

    if occupation == "Others":
        for occ in unique_occupations:
            input_data[f"Occupation_{occ}"] = [0]
    else:
        for occ in unique_occupations:
            input_data[f"Occupation_{occ}"] = [1 if occupation == occ else 0]

    required_columns = X.columns
    for col in required_columns:
        if col not in input_data.columns:
            input_data[col] = 0

    input_data[f'BMI Category_Normal'] = [1 if bmi_category == "Normal" else 0]
    input_data[f'BMI Category_Overweight'] = [1 if bmi_category == "Overweight" else 0]
    input_data[f'BMI Category_Obese'] = [1 if bmi_category == "Obese" else 0]

    input_data[f'Sleep Disorder_None'] = [1 if sleep_disorder == "None" else 0]
    input_data[f'Sleep Disorder_Sleep Apnea'] = [1 if sleep_disorder == "Sleep Apnea" else 0]
    input_data[f'Sleep Disorder_Insomnia'] = [1 if sleep_disorder == "Insomnia" else 0]

    input_data = input_data[X.columns]
    prediction = rf_model.predict(input_data)
    stress_level = prediction[0]

    if stress_level > 7:
        stress_label = "High"
        suggestion = ("High stress level detected. It's important to address your stress immediately. "
        "Consider practicing deep breathing exercises, mindfulness, or yoga to help manage stress. "
        "Regular physical exercise, such as walking, running, or cycling, can also reduce stress levels. "
        "Make sure you're getting enough sleep (7-9 hours per night) and maintaining a balanced diet. "
        "If these techniques do not help, consider seeking support from a counselor or therapist. "
        "Professional help can provide coping mechanisms to manage long-term stress.")
    elif 5 <= stress_level < 7:
        stress_label = "Medium"
        suggestion = (
        "Moderate stress level detected. This is a manageable level of stress, but you should still take steps "
        "to reduce it. Incorporating stress-relief activities into your daily routine, such as meditation, "
        "listening to calming music, or spending time with loved ones, can be helpful. "
        "Engaging in hobbies or activities you enjoy can serve as a positive outlet. "
        "Make sure you're taking breaks during work or study sessions to avoid burnout. "
        "Getting adequate sleep, staying hydrated, and maintaining regular exercise can further alleviate stress.")
    else:
        stress_label = "Low"
        suggestion = ("Low stress level detected. You're doing well at maintaining balance in your life! "
        "To keep your stress levels low, continue engaging in activities that promote relaxation, "
        "such as spending time outdoors, exercising regularly, and keeping up with healthy sleep habits. "
        "Stay connected with friends and family for emotional support and make time for hobbies you enjoy. "
        "It's also important to maintain a healthy diet and stay active to keep your mental and physical well-being in check.")

    return f"Predicted Stress Level: {stress_label} (Score: {round(stress_level, 2)})\n\nSuggestions: {suggestion}"

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
            
            # Process user input and get response
            response = process_user_input(user_input, st.session_state.user_data)
            
            # Add bot response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            st.rerun()

if __name__ == "__main__":
    main()
