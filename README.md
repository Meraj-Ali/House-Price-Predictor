# 🏡 House Price Predictor

A Streamlit web app for Exploratory Data Analysis (EDA) and House Price Prediction using machine learning.

### 📌 Features

**Sidebar navigation** with two sections:

**1.EDA** → Visualize the dataset with:

- Scatterplots

- Bar Plot

- Heatmap (correlations)

- Pie chart

- Count Plot

- Boxplot (distribution of price by category)

**2.Model & Prediction** → Train a regression model and predict house prices based on user inputs.

**Interactive inputs** with st.number_input (instead of sliders).

**Dynamic visualizations** using Matplotlib & Seaborn.

**ML model**: Linear Regression (easily extendable to others like Random Forest, XGBoost).

### 📂 Project Structure
house-price-predictor/

│── app.py                # Main entry point (links EDA & Prediction)

│── eda.py                # Exploratory Data Analysis page

│── model_prediction.py   # Model training & prediction page

│── enhanced_house_price_dataset.csv   # Dataset

│── requirements.txt      # Python dependencies


### 🚀 How to Run

##### Clone this repository or download the project folder.

git clone https://github.com/your-username/house-price-predictor.git
cd house-price-predictor


##### Install dependencies.

pip install -r requirements.txt


##### Run the Streamlit app.

streamlit run app.py

### 📊 Example Visualizations

- Heatmap of correlations

- Scatterplot of price vs square footage

- Pie chart of houses by city

- Boxplot of price vs furnishing status

- Barplot of Average Price by City
  
- Countplot of House count by furnishing status

### 🧠 Future Improvements

- Add multiple ML models (Random Forest, Gradient Boosting, etc.)

- Hyperparameter tuning

- Model evaluation metrics (R², MAE, RMSE)

- Save & load trained model with joblib

### 🛠️ Tech Stack

- Python

- Streamlit

- Pandas, NumPy

- Matplotlib, Seaborn

- Scikit-learn

👨‍💻 Author

Meraj Ali
