import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder

def load_model():
    with open("linear_regression_model.pkl", 'rb') as file:
        model,scaler,le = pickle.load(file)
    return model, scaler, le

def preprocessing_input_data(data,scaler,le):
    data['Extracurricular Activities'] = le.transform([data['Extracurricular Activities']])
    df=pd.DataFrame(data)
    df_tranformed = scaler.transform(df)
    return df_tranformed

def predict_data(data):
    model, scaler, le = load_model()
    processed_data=preprocessing_input_data(data,scaler,le)
    prediction=model.predict(processed_data)
    return prediction


def main():
    st.title("Student peformace prediction")
    st.write("Enter the details of the student to predict their performance score.")
    hour_studied = st.number_input('Hours Studied',min_value=1, max_value=10, value=5)
    previous_score = st.number_input('Previous Score',min_value=40,max_value=100)
    extracurricular_activities = st.selectbox('Extracurricular Activities', options=['Yes', 'No'])
    sleeping_hours = st.number_input('Sleeping Hours',min_value=4, max_value=10, value=7)
    question_papers_solved = st.number_input('Number of Question Papers Solved',min_value=0,max_value=10,value=5)
     
    if st.button('Predict Performance Score'):
        user_data={
            "Hours Studied":hour_studied,
            "Previous Scores":previous_score,
            "Extracurricular Activities":extracurricular_activities,
            "Sleep Hours":sleeping_hours,
            "Sample Question Papers Practiced":question_papers_solved,
        }
        prediction=predict_data(user_data)
        st.success(f"The predicted performance score of the student is: {prediction[0]:.2f}")
if __name__ == "__main__":
    main()