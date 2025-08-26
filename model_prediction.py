import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

@st.cache_data
def load_data():
    return pd.read_csv("enhanced_house_price_dataset.csv")

def show_page():
    data=load_data()

    st.title("🤖 Train Model & Predict Prices")

    st.header("🏡 Predict House Price")

    # Encode categorical variables
    encoded_data=pd.get_dummies(data, drop_first=True)
    X=encoded_data.drop("Price", axis=1)
    y=encoded_data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    model=RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # User inputs
    st.markdown('#### Enter House Details')
    input_area=st.number_input("Area (sqft)", min_value=500, max_value=10000, value=2000)
    input_bedrooms=st.number_input("Bedrooms", 1, 10, 3)
    input_bathrooms=st.number_input("Bathrooms", 1, 5, 2)
    input_stories=st.number_input("Stories", 1, 5, 2)
    input_parking=st.number_input("Parking Spaces", 0, 5, 1)
    input_age=st.number_input("Age of House (years)", 0, 50, 10)

    input_city=st.selectbox("City", data["City"].unique())
    input_furnishing=st.selectbox("Furnishing", data["Furnishing"].unique())
    input_water=st.selectbox("Water Supply", data["Water Supply"].unique())
    input_ac=st.selectbox("Air Conditioning", data["Air Conditioning"].unique())

    # Prepare input
    input_dict={
        "Area": input_area,
        "Bedrooms": input_bedrooms,
        "Bathrooms": input_bathrooms,
        "Stories": input_stories,
        "Parking": input_parking,
        "Age": input_age,
        "City": input_city,
        "Furnishing": input_furnishing,
        "Water Supply": input_water,
        "Air Conditioning": input_ac
    }

    input_df=pd.DataFrame([input_dict])
    input_encoded=pd.get_dummies(input_df)
    input_encoded=input_encoded.reindex(columns=X.columns, fill_value=0)

    if st.button("Predict Price"):
        prediction=model.predict(input_encoded)[0]
        st.success(f"Predicted House Price: ₹ {int(prediction):,}")