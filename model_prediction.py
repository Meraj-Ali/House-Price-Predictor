import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

@st.cache_data
def load_data():
    return pd.read_csv("enhanced_house_price_dataset.csv")

def show_page():
    data=load_data()

    st.title("🤖 Train Model & Predict Prices")

    st.header("🔮 Predict House Price")

    # Encode categorical variables
    encoded_data=pd.get_dummies(data, drop_first=True)
    X=encoded_data.drop("Price", axis=1)
    y=encoded_data["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

    # sidebar model selection
    st.sidebar.header("⚙️ Model Selection")
    model_choice=st.sidebar.selectbox(
        "Select Model",
        ("Linear Regression", "Random Forest", "Gradient Boosting")
    )

    # set parameter grids
    if model_choice=='Linear Regression': 
        model=LinearRegression()
        param_grid={}
    elif model_choice=='Random Forest':
        model=RandomForestRegressor(random_state=42)
        param_grid={
            'n_estimators':[50, 100, 200],
            'max_depth':[None, 5, 10]
        }
    else:
        model=GradientBoostingRegressor(random_state=42)
        param_grid={
            'n_estimators':[50, 100, 200],
            'learning_rate':[0.01, 0.1, 0.2],
            'max_depth':[3, 5, 7]
        }

    # Hyperparameter tuning if params exist
    if param_grid:
        grid=GridSearchCV(model, param_grid, cv=3, scoring='r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model=grid.best_estimator_
        st.sidebar.success(f"Best Params: {grid.best_params_}")
    else:
        best_model=model
        best_model.fit(X_train, y_train)

    # predictions
    y_pred=best_model.predict(X_test)

    # Model evaluation 
    st.sidebar.subheader("📊 Model Evaluation")
    st.sidebar.write(f"**R2 Score:** {r2_score(y_test, y_pred):.3f}")
    st.sidebar.write(f"**MAE:** { mean_absolute_error(y_test, y_pred):,.0f}")
    st.sidebar.write(f"**RMSE:** {np.sqrt(mean_squared_error(y_test, y_pred)):,.0f}")
    
    # model=RandomForestRegressor(n_estimators=100, random_state=42)
    # model.fit(X_train, y_train)

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
        prediction=best_model.predict(input_encoded)[0]

        st.success(f"Predicted House Price: ₹ {int(prediction):,}")
