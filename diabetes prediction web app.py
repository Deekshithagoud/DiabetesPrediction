# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 09:00:56 2024

@author: mdeek
"""

import numpy as np
import pickle
import streamlit as st

# loading the saved model
loaded_model = pickle.load(open('D:/Diabetes/trained_model.sav', 'rb'))

#creating a function for prediction

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'
  
    
def main():
    
    #giving a title 
    st.title("Diabetes prediction Web app")
    
    #getting the input data from the user
    
    Pregnancies = st.text_input("No. Of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood Pressure level")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin Level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function Value")
    Age = st.text_input("Age of the person")
    
    #code for prediction 
    diagnosis = ''
    
    #creating a button for prediction
    
    if st.button("Diabetes Test result"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
    
    st.success(diagnosis)
    
    
    
    
    
if __name__ == '__main__':
    main()
    
    