import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import traceback

# File uploader for CSV
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and prepare the dataset
    df_cleaned = pd.read_csv(uploaded_file)
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
    X = df_encoded.drop(columns=['Stress Level'])
    y = df_encoded['Stress Level']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Get unique occupations
    unique_occupations = df_cleaned['Occupation'].unique().tolist()
    unique_occupations.append("Others")

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
        .button-container {
            display: flex;
            gap: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Welcome to Mental Stress Manager")
    st.markdown("""
    This app is your personal AI companion for managing stress and mental wellbeing.

    ### Features:
    - Chat interface for stress assessment
    - AI-powered predictions
    - Personalized recommendations
    - Progress tracking
    """)

    def predict_stress(gender, age, occupation, sleep_duration, quality_of_sleep, activity_level, 
                    bmi_category, systolic_bp, diastolic_bp, heart_rate, daily_steps, sleep_disorder):
        
        input_data = pd.DataFrame({
            'Age': [age],
            'Sleep Duration': [sleep_duration],
            'Quality of Sleep': [quality_of_sleep],
            'Physical Activity Level': [activity_level],
            'Heart Rate': [heart_rate],
            'Daily Steps': [daily_steps],
            'Gender_Male': [1 if gender.lower() == "male" else 0],
            'Systolic_BP': [systolic_bp],
            'Diastolic_BP': [diastolic_bp]
        })

        # Handle occupation encoding
        if occupation == "Others":
            for occ in unique_occupations:
                input_data[f"Occupation_{occ}"] = [0]
        else:
            for occ in unique_occupations:
                input_data[f"Occupation_{occ}"] = [1 if occupation == occ else 0]

        # Encode BMI category
        input_data[f'BMI Category_Normal'] = [1 if bmi_category == "Normal" else 0]
        input_data[f'BMI Category_Overweight'] = [1 if bmi_category == "Overweight" else 0]
        input_data[f'BMI Category_Obese'] = [1 if bmi_category == "Obese" else 0]

        # Encode Sleep Disorder
        input_data[f'Sleep Disorder_None'] = [1 if sleep_disorder == "None" else 0]
        input_data[f'Sleep Disorder_Sleep Apnea'] = [1 if sleep_disorder == "Sleep Apnea" else 0]
        input_data[f'Sleep Disorder_Insomnia'] = [1 if sleep_disorder == "Insomnia" else 0]

        # Align columns with training data
        input_data = input_data[X.columns]
        
        # Make prediction
        prediction = rf_model.predict(input_data)[0]
        
        # Generate response based on stress level
        if prediction > 7:
            stress_label = "High"
            suggestion = ("High stress level detected. It's important to address your stress immediately. "
                        "Consider practicing deep breathing exercises, mindfulness, or yoga to help manage stress. "
                        "Regular physical exercise, such as walking, running, or cycling, can also reduce stress levels. "
                        "Make sure you're getting enough sleep (7-9 hours per night) and maintaining a balanced diet. "
                        "If these techniques do not help, consider seeking support from a counselor or therapist. "
                        "Professional help can provide coping mechanisms to manage long-term stress.")
        elif prediction > 4:
            stress_label = "Medium"
            suggestion = ("Moderate stress level detected. Here are some helpful tips:\n"
                        "1. Start a 10-minute daily meditation practice\n"
                        "2. Take regular breaks every 2 hours\n"
                        "3. Go for a 15-minute walk outside\n"
                        "4. Practice mindful breathing when feeling overwhelmed\n"
                        "5. Maintain a consistent sleep schedule")
        else:
            stress_label = "Low"
            suggestion = ("Your stress levels appear to be well-managed. Keep up the good work!\n"
                        "Tips to maintain this positive state:\n"
                        "1. Continue your current healthy routines\n"
                        "2. Stay physically active\n"
                        "3. Maintain good sleep habits\n"
                        "4. Practice gratitude daily\n"
                        "5. Stay connected with friends and family")
        
        return f"Predicted Stress Level: {stress_label} ({prediction:.1f}/10)\n\n{suggestion}"

    def process_user_input(message, user_data):
        try:
            message = message.strip()
            
            if "gender" not in user_data:
                user_data["gender"] = message.strip().lower()
                return "Great! Thank you for sharing. Could you please tell me your age?"
            
            # ... (rest of the process_user_input function remains the same)
            
        except Exception as e:
            print(traceback.format_exc())
            return "An error occurred while processing your input. Please try again."

    def main():
        st.markdown("<h1>Mental Stress Manager</h1>", unsafe_allow_html=True)
        
        # Chat interface
        user_input = st.text_input("Type your message...")
        
        if st.button("Send"):
            if user_input:
                # Process user input and get response
                response = process_user_input(user_input, {})
                
                # Display the conversation
                st.markdown(f"""
                    <div class='chat-message user-message'>
                        <div>{user_input}</div>
                    </div>
                    <div class='chat-message bot-message'>
                        <div>{response}</div>
                    </div>
                """, unsafe_allow_html=True)

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

else:
    st.error("Please upload a CSV file to proceed.")
