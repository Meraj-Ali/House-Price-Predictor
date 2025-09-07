import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

@st.cache_data
def load_data():
    return pd.read_csv("enhanced_house_price_dataset.csv")


def show_page():
    data=load_data()

    st.title("📊 Exploratory Data Analysis")

    with st.expander("Dataset Overview"):
        st.write("Shape:", data.shape)
        st.dataframe(data.head())

    # Sidebar filters
    st.sidebar.header("Filter Options:")
    city=st.sidebar.multiselect("Select City:", options=data['City'].unique(),default=data['City'].unique())
    bedrooms_min = st.sidebar.number_input("Min Bedrooms:", int(data["Bedrooms"].min()), int(data["Bedrooms"].max()), 1)
    bedrooms_max = st.sidebar.number_input("Max Bedrooms:", int(data["Bedrooms"].min()), int(data["Bedrooms"].max()), 5)
    furnishing = st.sidebar.multiselect("Furnishing:", options=data["Furnishing"].unique(), default=data["Furnishing"].unique())

    filtered_data=data[
        (data["City"].isin(city)) & 
        (data["Bedrooms"] >= bedrooms_min) & (data["Bedrooms"] <= bedrooms_max) &
        (data["Furnishing"].isin(furnishing))
    ]

    with st.expander("Filtered Data"):
        st.write(filtered_data.shape)
        st.dataframe(filtered_data.head())

    # Scatter plot
    with st.expander("Scatter Plot: Area vs Price"):
        fig, ax=plt.subplots()
        sns.scatterplot(data=filtered_data, x="Area", y="Price", hue="City", ax=ax)
        ax.set_title("Area vs Price")
        st.pyplot(fig)

    # Bar plot
    with st.expander("Bar Plot: Average Price by City"):
        fig, ax=plt.subplots()
        avg_price_city=filtered_data.groupby("City")["Price"].mean().reset_index()
        sns.barplot(data=avg_price_city, x="City", y='Price', hue='City', ax=ax)
        ax.set_title("Average Price by City")
        st.pyplot(fig)

    # Price Distribution
    with st.expander("Distribution Plot: House Prices"):
        fig, ax = plt.subplots()
        sns.histplot(filtered_data["Price"], bins=30, kde=True, ax=ax)
        ax.set_title("Price Distribution")
        st.pyplot(fig)

    # Log-Price Distribution
    with st.expander("Log-Transformed Price Distribution"):
        fig, ax = plt.subplots()
        sns.histplot(np.log1p(filtered_data["Price"]), bins=30, kde=True, ax=ax)
        ax.set_title("Log(Price) Distribution (Better for Regression)")
        st.pyplot(fig)

    # Price vs Bedrooms
    with st.expander("Box Plot: Price vs Bedrooms"):
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_data, x="Bedrooms", y="Price", ax=ax)
        ax.set_title("Price Distribution across Bedroom Counts")
        st.pyplot(fig)

    # Price vs Bedrooms (Violin Plot for richer detail)
    with st.expander("Violin Plot: Price vs Bedrooms"):
        fig, ax = plt.subplots()
        sns.violinplot(data=filtered_data, x="Bedrooms", y="Price", ax=ax)
        ax.set_title("Price vs Bedrooms (Violin Plot)")
        st.pyplot(fig)

    # Pairplot (Selected Features)
    with st.expander("Pairplot: Feature Relationships"):
        selected_features = ["Area", "Bedrooms", "Price"]
        sns_plot = sns.pairplot(filtered_data[selected_features], diag_kind="kde")
        st.pyplot(sns_plot)

    # FacetGrid: Bedrooms vs Price across Cities
    with st.expander("FacetGrid: Bedrooms vs Price across Cities"):
        g = sns.FacetGrid(filtered_data, col="City", col_wrap=3, height=4)
        g.map(sns.boxplot, "Bedrooms", "Price")
        st.pyplot(g)

    # Heatmap
    with st.expander("Heatmap: Feature Correlation"):
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation")
        st.pyplot(fig)
