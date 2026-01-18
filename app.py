import gradio as gr
import pandas as pd
import pickle
import numpy as np

with open("best_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict_diatetics(
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI,DiabetesPedigreeFunction,Age
):
    input_df = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin,BMI,DiabetesPedigreeFunction,Age

    ]],
      columns=[
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
       'BMI', 'DiabetesPedigreeFunction', 'Age'
    ])
    
    prediction = model.predict(input_df)[0]

    return "Diabetes Detected" if prediction == 1 else "No Diabetes"


inputs=[
        gr.Number(label="Pregnancies (Number of times pregnant)"),
        gr.Number(label="Glucose (Plasma glucose concentration a 2 hours in an oral glucose tolerance test)"),
        gr.Number(label="Blood Pressure (Diastolic blood pressure (mm Hg))"),
        gr.Number(label="Skin Thickness (Triceps skin fold thickness (mm))"),
        gr.Number(label="Insulin (2-Hour serum insulin (mu U/ml))"),
        gr.Number(label="BMI (Body mass index (weight in kg/(height in m)^2))"),
        gr.Number(label="Diabetes Pedigree Function"),
        gr.Number(label="Age (years)")
    ]

app = gr.Interface(
    fn=predict_diatetics,
    inputs=inputs,
    outputs="text",
    title="Diabetes Prediction App"
)

app.launch()
