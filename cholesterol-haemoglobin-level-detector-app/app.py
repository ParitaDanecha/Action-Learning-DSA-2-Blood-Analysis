from scipy.signal import savgol_filter
import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.express as px
import json

def snv(input_data):
    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:])
    return output_data

def savgol(x):    
    return savgol_filter(x, 5, 3)

header = st.container()
with header:
        
    st.sidebar.markdown("<h1 style='text-align: left; color: dark-violet;'>Choose Your Option(s)</h1>", unsafe_allow_html=True)
    hdl_cholesterol = st.sidebar.checkbox('HDL Cholesterol')
    cholesterol_ldl = st.sidebar.checkbox('Cholesterol LDL')
    hemoglobin = st.sidebar.checkbox('Hemoglobin')

    st.sidebar.markdown("<h3 style='text-align: left; color: dark-violet;'>Preprocessing (Optional)</h3>", unsafe_allow_html=True)
    SNV = st.sidebar.checkbox('SNV')
    SAVGOL = st.sidebar.checkbox('SAVGOL')

    st.markdown("<h1 style='text-align: center; color: dark-violet;'>Prediction of Cholestrol and Haemoglobin Levels in Blood Samples</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        dataframe = pd.read_csv(uploaded_file)
        df_json = dataframe.to_json(orient="values")
        # st.write(dataframe.head(5))
        df_display = dataframe.copy()

        if SNV:
            X = df_display.values[:,0:170]
            ans=snv(X)
            df_display.iloc[:, 0:170] = ans

        if SAVGOL:
            X = df_display.values[:,0:170]
            ans = np.apply_along_axis(savgol, 1, X)
            df_display.iloc[:, 0:170] = ans

        df_t = df_display.T 
        cols = ['Patient_'+str(i) for i in range(df_t.shape[1])]
        df_t.columns = cols
        # st.write(df_t)

        st.markdown("<h4 style='text-align: center; color: dark-violet;'>Visualization of Patient Samples</h4>", unsafe_allow_html=True)
        a = np.linspace(900, 1700, 170)
        df_t["wavelength"] = a
        series = df_t.columns
        fig = px.line(df_t, x="wavelength", y=series)   
        st.write(fig)    

    if st.sidebar.button("Predict"): 
        st.markdown("<h4 style='text-align: center; color: dark-violet;'>Prediction of Patient Samples</h4>", unsafe_allow_html=True)           
        if hdl_cholesterol:
            url = "https://hdl-cholesterol.herokuapp.com/predictHdlCholesterol" 
            data = {'data': str(df_json)}
            prediction_hdl = requests.post(url, json=data)
            dataframe['prediction_hdl'] = prediction_hdl.json()
            
        if cholesterol_ldl:
            url = "https://cholesterol-ldl.herokuapp.com/predictCholesterolLdl"
            data = {'data': str(df_json)}
            prediction_ldl = requests.post(url, json=data)
            dataframe['prediction_ldl'] = prediction_ldl.json()

        if hemoglobin:
            url = "https://haemoglobin.herokuapp.com/predictHemoglobin"
            data = {'data': str(df_json)}
            prediction_hgb = requests.post(url, json=data)
            dataframe['prediction_hgb'] = prediction_hgb.json()

        for i in range(len(dataframe)):
            series = dataframe.iloc[i,:170]            
            fig = px.line(x=a, y=series, title="Patient "+str(i))
            fig.update_layout(showlegend=False)
            st.write(fig)
            result= []
            if hdl_cholesterol:
                result.append("HDL Cholesterol Level: "+dataframe['prediction_hdl'][i])

            if cholesterol_ldl:
                result.append("LDL Cholesterol Level: "+dataframe['prediction_ldl'][i])

            if hemoglobin:
                result.append("Hemoglobin Level: "+dataframe['prediction_hgb'][i])

            result = ", ".join(result)
            st.markdown(f"<p style='text-align: center;'>{result}</p>", unsafe_allow_html=True)
    