import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler,LabelEncoder
import os
import pickle
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi


uri = "mongodb+srv://<username>:<password>@cluster0.ma6xrby.mongodb.net/?appName=Cluster0"
client = MongoClient(uri, server_api=ServerApi('1'))
db=client['student']
collection=db['student_pred']


def load_model():
    base_dir = os.path.dirname(__file__)  # directory of student_perf.py
    model_path = os.path.join(base_dir, "linear_regression_model.pkl")

    with open(model_path, "rb") as file:
        model, scaler, le = pickle.load(file)

    return model, scaler, le



def preprocessing_input_data(data, scaler, le):
    data['Extracurricular Activities'] = int(
        le.transform([data['Extracurricular Activities']])[0]
    )
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed


def predict_data(data):
    model, scaler, le = load_model()

    # Copy to avoid mutating original dict
    data_copy = data.copy()

    processed_data = preprocessing_input_data(data_copy, scaler, le)
    prediction = model.predict(processed_data)

    return float(prediction[0])  # convert to Python float



def main():
    st.title("Student peformace prediction")
    st.write("Enter the details of the student to predict their performance score.")
    hour_studied = st.number_input('Hours Studied',min_value=1, max_value=10, value=5)
    previous_score = st.number_input('Previous Score',min_value=40,max_value=100)
    extracurricular_activities = st.selectbox('Extracurricular Activities', options=['Yes', 'No'])
    sleeping_hours = st.number_input('Sleeping Hours',min_value=4, max_value=10, value=7)
    question_papers_solved = st.number_input('Number of Question Papers Solved',min_value=0,max_value=10,value=5)
     
    if st.button('Predict Performance Score'):
        user_data = {
            "Hours Studied": int(hour_studied),
            "Previous Scores": int(previous_score),
            "Extracurricular Activities": extracurricular_activities,  # keep original
            "Sleep Hours": int(sleeping_hours),
            "Sample Question Papers Practiced": int(question_papers_solved),
        }

        prediction = predict_data(user_data)

        st.success(f"The predicted performance score of the student is: {prediction:.2f}")

        # Mongo-safe document
        mongo_data = user_data.copy()
        mongo_data["prediction"] = prediction

        collection.insert_one(mongo_data)

if __name__ == "__main__":
    main()