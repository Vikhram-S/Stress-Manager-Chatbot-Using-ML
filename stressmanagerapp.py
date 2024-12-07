import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import streamlit as st
import traceback

# Set page configuration first
st.set_page_config(
    page_title="Mental Stress Manager", 
    layout="wide"
)

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None

# File uploader for CSV
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Load and prepare the dataset
    df_cleaned = pd.read_csv(uploaded_file)
    df_encoded = pd.get_dummies(df_cleaned, drop_first=True)
    X = df_encoded.drop(columns=['Stress Level'])
    y = df_encoded['Stress Level']

    # Store feature columns in session state
    st.session_state.feature_columns = X.columns

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForest model and store in session state
    rf_model = RandomForestRegressor(random_state=42)
    rf_model.fit(X_train, y_train)
    st.session_state.model = rf_model
    
    # Get unique occupations
    unique_occupations = df_cleaned['Occupation'].unique().tolist()
    unique_occupations.append("Others")

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
            border: 1px solid #4ECDC4;
            border-radius: 15px;
            margin-bottom: 20px;
            background: rgba(255,255,255,0.05);
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
        .advice-box {
            padding: 20px;
            border-radius: 15px;
            background: linear-gradient(45deg, #2C3E5022, #3498DB22);
            margin: 20px 0;
            border: 1px solid #3498DB;
        }
        .footer {
            background: linear-gradient(45deg, #2C3E50, #3498DB);
            padding: 30px;
            border-radius: 20px;
            margin-top: 50px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("Welcome to Mental Stress Manager")
    st.markdown("""
    This app is your personal AI companion for managing stress and mental wellbeing.

    ### Features:
    - Structured questionnaire for stress assessment
    - AI-powered predictions
    - Detailed personalized recommendations
    - Holistic wellness advice
    """)

    def get_detailed_advice(stress_level, user_data):
        age = int(user_data['age'])
        sleep_quality = int(user_data['sleep_quality'])
        activity_level = int(user_data['activity_level'])
        
        advice = {
            "High": {
                "Immediate Actions": [
                    "Take deep breaths for 5 minutes every hour",
                    "Step away from stressful situations when possible",
                    "Practice the 5-4-3-2-1 grounding technique"
                ],
                "Daily Practices": [
                    f"Given your activity level of {activity_level}/10, gradually increase physical activity",
                    f"With your sleep quality at {sleep_quality}/10, focus on sleep hygiene",
                    "Maintain a stress journal to identify triggers",
                    "Practice progressive muscle relaxation before bed"
                ],
                "Long-term Strategies": [
                    "Consider professional counseling or therapy",
                    "Join stress management workshops",
                    "Build a support network",
                    "Learn time management techniques"
                ],
                "Lifestyle Modifications": [
                    "Reduce caffeine and processed foods",
                    "Create a calming morning routine",
                    "Set boundaries in work and personal life",
                    "Take up a relaxing hobby like gardening or painting"
                ]
            },
            "Medium": {
                "Daily Practices": [
                    "15-minute morning meditation",
                    f"Based on your age ({age}), appropriate exercise routine",
                    "Regular breaks during work",
                    "Nature walks or outdoor time"
                ],
                "Wellness Tips": [
                    "Practice mindful eating",
                    "Maintain a gratitude journal",
                    "Regular stretching exercises",
                    "Digital detox for 1 hour before bed"
                ],
                "Preventive Measures": [
                    "Set realistic goals and priorities",
                    "Create a balanced weekly schedule",
                    "Practice saying 'no' when necessary",
                    "Regular social connections"
                ]
            },
            "Low": {
                "Maintenance Tips": [
                    "Continue your effective stress management practices",
                    "Regular exercise and movement",
                    "Maintain social connections",
                    "Healthy sleep schedule"
                ],
                "Enhancement Strategies": [
                    "Set new personal growth goals",
                    "Learn new skills or hobbies",
                    "Share your successful strategies with others",
                    "Regular wellness check-ins"
                ]
            }
        }
        return advice[stress_level]

    def predict_stress(user_data):
        if st.session_state.model is None or st.session_state.feature_columns is None:
            st.error("Please upload dataset first")
            return None, None

        input_data = pd.DataFrame({
            'Age': [user_data['age']],
            'Sleep Duration': [user_data['sleep_duration']],
            'Quality of Sleep': [user_data['sleep_quality']],
            'Physical Activity Level': [user_data['activity_level']],
            'Heart Rate': [user_data['heart_rate']],
            'Daily Steps': [user_data['daily_steps']],
            'Gender_Male': [1 if user_data['gender'].lower().strip() == "male" else 0],
            'Systolic_BP': [user_data['systolic_bp']],
            'Diastolic_BP': [user_data['diastolic_bp']]
        })

        # Handle occupation encoding
        if user_data['occupation'].strip().title() == "Others":
            for occ in unique_occupations:
                input_data[f"Occupation_{occ}"] = [0]
        else:
            for occ in unique_occupations:
                input_data[f"Occupation_{occ}"] = [1 if user_data['occupation'].strip().title() == occ else 0]

        # Encode BMI category
        input_data[f'BMI Category_Normal'] = [1 if user_data['bmi_category'].strip().title() == "Normal" else 0]
        input_data[f'BMI Category_Overweight'] = [1 if user_data['bmi_category'].strip().title() == "Overweight" else 0]
        input_data[f'BMI Category_Obese'] = [1 if user_data['bmi_category'].strip().title() == "Obese" else 0]

        # Encode Sleep Disorder
        input_data[f'Sleep Disorder_None'] = [1 if user_data['sleep_disorder'].strip().title() == "None" else 0]
        input_data[f'Sleep Disorder_Sleep Apnea'] = [1 if user_data['sleep_disorder'].strip().title() == "Sleep Apnea" else 0]
        input_data[f'Sleep Disorder_Insomnia'] = [1 if user_data['sleep_disorder'].strip().title() == "Insomnia" else 0]

        # Reorder columns to match training data
        input_data = input_data.reindex(columns=st.session_state.feature_columns, fill_value=0)
        
        prediction = st.session_state.model.predict(input_data)[0]
        
        if prediction > 7:
            return "High", prediction
        elif prediction > 4:
            return "Medium", prediction
        else:
            return "Low", prediction

    def main():
        st.markdown("<h1>Mental Stress Assessment</h1>", unsafe_allow_html=True)
        
        questions = [
            ("Please enter your gender (Male/Female):", "gender", lambda x: x.strip().lower() in ["male", "female"]),
            ("What is your age?", "age", lambda x: x.strip().isdigit() and 18 <= int(x) <= 100),
            (f"What is your occupation? ({', '.join(unique_occupations)}):", "occupation", lambda x: x.strip().title() in unique_occupations),
            ("How many hours do you sleep per day? (4-12)", "sleep_duration", lambda x: x.strip().replace('.','',1).isdigit() and 4 <= float(x) <= 12),
            ("Rate your sleep quality (1-10):", "sleep_quality", lambda x: x.strip().isdigit() and 1 <= int(x) <= 10),
            ("Rate your physical activity level (1-10):", "activity_level", lambda x: x.strip().isdigit() and 1 <= int(x) <= 10),
            ("Enter your BMI category (Normal/Overweight/Obese):", "bmi_category", lambda x: x.strip().title() in ["Normal", "Overweight", "Obese"]),
            ("Enter your systolic blood pressure (90-200):", "systolic_bp", lambda x: x.strip().isdigit() and 90 <= int(x) <= 200),
            ("Enter your diastolic blood pressure (60-130):", "diastolic_bp", lambda x: x.strip().isdigit() and 60 <= int(x) <= 130),
            ("Enter your heart rate (60-120):", "heart_rate", lambda x: x.strip().isdigit() and 60 <= int(x) <= 120),
            ("Enter your daily steps (1000-20000):", "daily_steps", lambda x: x.strip().isdigit() and 1000 <= int(x) <= 20000),
            ("Do you have any sleep disorder? (None/Sleep Apnea/Insomnia):", "sleep_disorder", lambda x: x.strip().title() in ["None", "Sleep Apnea", "Insomnia"])
        ]

        # Display chat history
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"""
                    <div class='chat-container'>
                        <div class='chat-message bot-message'>{q}</div>
                        <div class='chat-message user-message'>{a}</div>
                    </div>
                """, unsafe_allow_html=True)

        if st.session_state.step < len(questions):
            question, key, validator = questions[st.session_state.step]
            user_input = st.text_input(question, key=f"input_{st.session_state.step}")
            
            col1, col2 = st.columns(2)
            with col1:
                next_button = st.button("Next", key="next_button")
                if next_button and not st.session_state.get('next_clicked', False):
                    st.session_state.next_clicked = True
                    if validator(user_input):
                        st.session_state.user_data[key] = user_input.strip()
                        st.session_state.chat_history.append((question, user_input.strip()))
                        st.session_state.step += 1
                    else:
                        st.error("Please provide a valid input")
                    st.session_state.next_clicked = False
            with col2:
                clear_button = st.button("Clear", key="clear_button")
                if clear_button and not st.session_state.get('clear_clicked', False):
                    st.session_state.clear_clicked = True
                    st.session_state.step = 0
                    st.session_state.user_data = {}
                    st.session_state.chat_history = []
                    st.session_state.clear_clicked = False
                    
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Get Detailed Assessment"):
                    level, score = predict_stress(st.session_state.user_data)
                    if level is not None and score is not None:
                        detailed_advice = get_detailed_advice(level, st.session_state.user_data)
                        
                        st.markdown(f"""
                            <div class='advice-box'>
                                <h2>Your Stress Assessment</h2>
                                <h3>Stress Level: {level} ({score:.1f}/10)</h3>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        for category, tips in detailed_advice.items():
                            st.markdown(f"""
                                <div class='advice-box'>
                                    <h3>{category}</h3>
                                    <ul>
                                        {"".join([f"<li>{tip}</li>" for tip in tips])}
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
            with col2:
                if st.button("Start Over"):
                    st.session_state.step = 0
                    st.session_state.user_data = {}
                    st.session_state.chat_history = []
                    

        # Footer
        st.markdown("""
            <div class='footer'>
                <h3>Meet Our Exceptional Development Team</h3>
                <div class='team-section'>
                    <p><strong>Project Lead Developer</strong><br>Vikhram S</p>
                    <p><strong>Co-Developers</strong><br>
                    • Ragul S<br>
                    • Roshan R<br>
                    • Nithesh Kumar B</p>
                </div>
                <p>© 2024 Mental Stress Manager by Z Data Knights. All Rights Reserved.</p>
                <p style='text-align: center; margin-top: 15px;'>Made With ❤️ by Team Z Data Knights</p>
            </div>
        """, unsafe_allow_html=True)

    if __name__ == "__main__":
        main()

else:
    st.error("Please upload a CSV file to proceed.")
