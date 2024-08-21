import pickle
import pandas as pd
import streamlit as st

# Set the title of the app
st.title("❤️ Predictive Modelling for Cardiovascular Diagnosis using Machine Learning ❤️")
st.header("Input Patient's Details")

# Function to load the trained model pipeline
@st.cache_resource
def load_model_pipeline(model_path):
    with open(model_path, 'rb') as file:
        model_pipeline = pickle.load(file)
    return model_pipeline

# Function to predict from user input using the model pipeline
def predict_from_user_input(model_pipeline, user_input_df):
    predicted_outcome = model_pipeline.predict(user_input_df)[0]
    return predicted_outcome

# Path to the model pipeline
model_path = 'D:/project/ML Models/model/model/Cardivascular_disease_best_model.pkl'  # Ensure this path matches your actual model file
model_pipeline = load_model_pipeline(model_path)

# Function to collect user inputs
def get_user_input():
    # Input age
    Age = st.number_input("Age", min_value=1, step=1, value=55)

    # Creating Columns for Chest pain, Shortness of breath, and Fatigue
    col1, col2, col3 = st.columns(3)

    # Chest pain
    with col1:
        st.write("Chest Pain?")
        chest_pain = st.checkbox("Yes", key="chest_pain")

    # Shortness of breath
    with col2:
        st.write("Shortness of breath?")
        shortness_of_breath = st.checkbox("Yes", key="shortness_of_breath")

    # Fatigue
    with col3:
        st.write("Fatigue?")
        fatigue = st.checkbox("Yes", key="fatigue")

    # Systolic
    Systolic = st.number_input("Systolic Pressure", min_value=1, step=1, value=140)

    # Diastolic
    Diastolic = st.number_input("Diastolic Pressure", min_value=1, step=1, value=90)

    # Heart rate (bpm)
    Heart_rate = st.number_input("Heart Rate (bpm)", min_value=1, step=1, value=100)

    # Lung sounds
    st.write("Lung Sounds?")
    lung_sounds = st.checkbox("Yes", key="lung_sounds")

    # Cholesterol level (mg/dL)
    cholesterol_level = st.number_input("Cholesterol level (mg/dL)", min_value=1, step=1, value=220)

    # LDL level (mg/dL)
    ldl_level = st.number_input("LDL level (mg/dL)", min_value=1, step=1, value=150)

    # HDL level (mg/dL)
    hdl_level = st.number_input("HDL level (mg/dL)", min_value=1, step=1, value=40)

    # Creating Columns for Diabetes, Atrial fibrillation, and Rheumatic fever
    col4, col5, col6 = st.columns(3)

    # Diabetes
    with col4:
        st.write("Diabetes?")
        diabetes = st.checkbox("Yes", key="diabetes")

    # Atrial fibrillation
    with col5:
        st.write("Atrial fibrillation?")
        atrial_fibrillation = st.checkbox("Yes", key="atrial_fibrillation")

    # Rheumatic fever
    with col6:
        st.write("Rheumatic fever?")
        rheumatic_fever = st.checkbox("Yes", key="rheumatic_fever")

    # Creating Columns for Mitral stenosis, Aortic stenosis, and Tricuspid stenosis
    col7, col8, col9 = st.columns(3)

    # Mitral stenosis
    with col7:
        st.write("Mitral stenosis?")
        mitral_stenosis = st.checkbox("Yes", key="mitral_stenosis")

    # Aortic stenosis
    with col8:
        st.write("Aortic stenosis?")
        aortic_stenosis = st.checkbox("Yes", key="aortic_stenosis")

    # Tricuspid stenosis
    with col9:
        st.write("Tricuspid stenosis?")
        tricuspid_stenosis = st.checkbox("Yes", key="tricuspid_stenosis")

    # Creating Columns for Pulmonary stenosis, Dilated cardiomyopathy, and Hypertrophic cardiomyopathy
    col10, col11, col12 = st.columns(3)

    # Pulmonary stenosis
    with col10:
        st.write("Pulmonary stenosis?")
        pulmonary_stenosis = st.checkbox("Yes", key="pulmonary_stenosis")

    # Dilated cardiomyopathy
    with col11:
        st.write("Dilated cardiomyopathy?")
        dilated_cardiomyopathy = st.checkbox("Yes", key="dilated_cardiomyopathy")

    # Hypertrophic cardiomyopathy
    with col12:
        st.write("Hypertrophic cardiomyopathy?")
        hypertrophic_cardiomyopathy = st.checkbox("Yes", key="hypertrophic_cardiomyopathy")

    # Creating Columns for Drug use, Fever, and Chills
    col13, col14, col15 = st.columns(3)

    # Drug use
    with col13:
        st.write("Drug use?")
        drug_use = st.checkbox("Yes", key="drug_use")

    # Fever
    with col14:
        st.write("Fever?")
        fever = st.checkbox("Yes", key="fever")

    # Chills
    with col15:
        st.write("Chills?")
        chills = st.checkbox("Yes", key="chills")

    # Creating Columns for Alcoholism, Hypertension, and Fainting
    col16, col17, col18 = st.columns(3)

    # Alcoholism
    with col16:
        st.write("Alcoholism?")
        alcoholism = st.checkbox("Yes", key="alcoholism")

    # Hypertension
    with col17:
        st.write("Hypertension?")
        hypertension = st.checkbox("Yes", key="hypertension")

    # Fainting
    with col18:
        st.write("Fainting?")
        fainting = st.checkbox("Yes", key="fainting")

    # Creating Columns for Dizziness, Smoking, and Obesity
    col19, col20, col21 = st.columns(3)

    # Dizziness
    with col19:
        st.write("Dizziness?")
        dizziness = st.checkbox("Yes", key="dizziness")

    # Smoking
    with col20:
        st.write("Smoking?")
        smoking = st.checkbox("Yes", key="smoking")

    # Obesity
    with col21:
        st.write("Obesity?")
        obesity = st.checkbox("Yes", key="obesity")

    # Murmur
    st.write("Murmur?")
    murmur = st.checkbox("Yes", key="murmur")

    # Convert user input into a dictionary
    user_input = {
        'Age': Age,
        'Chest pain': int(chest_pain),
        'Shortness of breath': int(shortness_of_breath),
        'Fatigue': int(fatigue),
        'Systolic': Systolic,
        'Diastolic': Diastolic,
        'Heart rate (bpm)': Heart_rate,
        'Lung sounds': int(lung_sounds),
        'Cholesterol level (mg/dL)': cholesterol_level,
        'LDL level (mg/dL)': ldl_level,
        'HDL level (mg/dL)': hdl_level,
        'Diabetes': int(diabetes),
        'Atrial fibrillation': int(atrial_fibrillation),
        'Rheumatic fever': int(rheumatic_fever),
        'Mitral stenosis': int(mitral_stenosis),
        'Aortic stenosis': int(aortic_stenosis),
        'Tricuspid stenosis': int(tricuspid_stenosis),
        'Pulmonary stenosis': int(pulmonary_stenosis),
        'Dilated cardiomyopathy': int(dilated_cardiomyopathy),
        'Hypertrophic cardiomyopathy': int(hypertrophic_cardiomyopathy),
        'Drug use': int(drug_use),
        'Fever': int(fever),
        'Chills': int(chills),
        'Alcoholism': int(alcoholism),
        'Hypertension': int(hypertension),
        'Fainting': int(fainting),
        'Dizziness': int(dizziness),
        'Smoking': int(smoking),
        'Obesity': int(obesity),
        'Murmur': int(murmur)
    }
    
    return pd.DataFrame(user_input, index=[0])

# Collect user input
user_input_df = get_user_input()

# Predict the outcome
if st.button('Predict'):
    predicted_outcome = predict_from_user_input(model_pipeline, user_input_df)
    st.write(f"The predicted cardiovascular disease is: **{predicted_outcome}**")
