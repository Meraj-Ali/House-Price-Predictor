import streamlit as st
import eda
import model_prediction

st.set_page_config(page_title="House Price Predictor", layout='wide')
# Sidebar 
st.sidebar.header("🏡 House Price Predictor")
page=st.sidebar.radio("Let's explore:", ["📊 EDA", "🤖 Model & Prediction"])


if page=="📊 EDA":
    eda.show_page()

elif page=="🤖 Model & Prediction":
    model_prediction.show_page()
