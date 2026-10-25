#--- all files save it in one folder only ---
#--- first pipeline basic.ipynb file run it---
#--- second refresh the folder where you have save all file ---
#--- third run the loading pickile file.ipynb ---
#--- fourth run the app.py file ---


#----  import module ---

import streamlit as st
import pandas as pd
import pickle


#--------------Page configuration-------

st.set_page_config(page_title="Diamond Price Predictor", layout="centered")


#----------Load trained pipeline---------

# read from a cache memory
@st.cache_resource


def load_model():
    with open("diamond_price_prediction.pkl","rb") as f:
        return pickle.load(f)

# loading the pipeline
model=load_model()


# ------- App title----------
# page design
st.title("💎 Diamond Price predictor app")
st.write("Predict diamond price using trained ML model")



# ------User inputs--------

carat=st.number_input("Carat",min_value=0.1,max_value=5.0,value=0.7)
cut=st.selectbox("Cut",["Fair","Good","Veru Good","Premium","Ideal"])
color=st.selectbox("Color",["D","E","F","I","J"])
clarity=st.selectbox("Clarity",["I1","SI2","SI1","VS2","VS1","VVS2","VVS1","IF"])
depth=st.number_input("depth",value=61.5)
table=st.number_input("Table",value=55.0)
x=st.number_input("X length in mm",value=5.7)
y=st.number_input("Y length in mm",value=5.7)
z=st.number_input("Z length in mm",value=5.7)


#-----Prediction-----------

if st.button("Predict price 💸"):
    input_df=pd.DataFrame([{
        "carat":carat,
        "cut":cut,
        "color":color,
        "clarity":clarity,
        "depth":depth,
        "table":table,
        "x":x,
        "y":y,
        "z":z}])


    prediction=model.predict(input_df)[0]   
    st.success(f"Estimated Diamond price is:{round(prediction,2)}") # toaster success message  


# --- to run this file 
#Click the Windows Start button.
#Type Anaconda Prompt.
#Click Anaconda Prompt when it appears.
#cd "C:\Users\ruqua\OneDrive\Desktop\Model Deployment"
#streamlit run app.py   => code to run a this file
# in browser it will open
# if it is not open then in promd ctrl+click it will open








