import streamlit as st
import pandas as pd
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
        sns.barplot(data=avg_price_city, x="City", y='Price', ax=ax)
        ax.set_title("Average Price by City")
        st.pyplot(fig)

    # Heatmap
    with st.expander("Heatmap: Feature Correlation"):
        fig, ax=plt.subplots(figsize=(10, 6))
        sns.heatmap(data.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation")
        st.pyplot(fig)

    # Pie Chart
    with st.expander("Pie Chart:Distribution of Houses by city"):
        city_counts=filtered_data["City"].value_counts()
        fig, ax=plt.subplots()
        ax.pie(
            city_counts,
            labels=city_counts.index,
            autopct="%1.1f%%",
            startangle=90,
            counterclock=False
        )
        ax.set_title("Houses by City")
        st.pyplot(fig)

    # Count Plot
    with st.expander("Count Plot: House Count by Furnishing Status"):
        fig, ax = plt.subplots()
        sns.countplot(data=filtered_data, x="Furnishing", ax=ax)
        ax.set_title("House Count by Furnishing Status")
        st.pyplot(fig)

    # Box plot
    with st.expander("Box Plot: Price Distribution by Furnishing"):
        fig, ax = plt.subplots()
        sns.boxplot(data=filtered_data, x="Furnishing", y="Price", ax=ax)
        ax.set_title("Price Distribution by Furnishing")
        st.pyplot(fig)
