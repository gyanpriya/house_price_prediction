import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Set page layout
st.set_page_config(page_title="House Price Dashboard", layout="wide")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv(r"C:\Users\hp\Desktop\House_price_prediction\data\Housing.csv")

# Load model and preprocessor
@st.cache_resource
def load_model_and_preprocessor():
    model = pickle.load(open("models/model.pkl", "rb"))
    preprocessor = pickle.load(open("models/encoder_scaler.pkl", "rb"))
    return model, preprocessor["encoder"], preprocessor["scaler"]

df = load_data()
model, encoder, scaler = load_model_and_preprocessor()

# Sidebar for navigation
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", ["🏠 EDA", "🔮 Prediction"])

# -----------------------------
# 🏠 EDA SECTION
# -----------------------------
if section == "🏠 EDA":
    st.title("🏠 Exploratory Data Analysis")
    
    st.subheader("Dataset Overview")
    st.dataframe(df.head())

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Missing Values")
    st.write(df.isnull().sum())

    st.subheader("Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df['price'], kde=True, ax=ax1, color="skyblue")
    st.pyplot(fig1)

    st.subheader("Correlation Heatmap")
    fig2, ax2 = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax2)
    st.pyplot(fig2)

    st.subheader("Scatter Plot: Area vs Price")
    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x='area', y='price', hue='furnishingstatus', ax=ax3)
    st.pyplot(fig3)

# -----------------------------
# 🔮 PREDICTION SECTION
# -----------------------------
elif section == "🔮 Prediction":
    st.title("🔮 Predict House Price")

    st.markdown("Fill the details in the sidebar to predict the house price.")

    # --- Sidebar inputs for all features used in model ---
    area = st.sidebar.slider("Area (sq ft)", 500, 10000, 1500)
    bedrooms = st.sidebar.selectbox("Bedrooms", [1, 2, 3, 4, 5])
    bathrooms = st.sidebar.selectbox("Bathrooms", [1, 2, 3, 4])
    stories = st.sidebar.selectbox("Stories", [1, 2, 3, 4])
    parking = st.sidebar.selectbox("Parking Spaces", [0, 1, 2, 3])

    mainroad = st.sidebar.selectbox("Main Road", ['yes', 'no'])
    guestroom = st.sidebar.selectbox("Guest Room", ['yes', 'no'])
    basement = st.sidebar.selectbox("Basement", ['yes', 'no'])
    hotwaterheating = st.sidebar.selectbox("Hot Water Heating", ['yes', 'no'])
    airconditioning = st.sidebar.selectbox("Air Conditioning", ['yes', 'no'])
    prefarea = st.sidebar.selectbox("Preferred Area", ['yes', 'no'])
    furnishingstatus = st.sidebar.selectbox("Furnishing Status", ['furnished', 'semi-furnished', 'unfurnished'])

    # --- Build input DataFrame just like training data ---
    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "mainroad": mainroad,
        "guestroom": guestroom,
        "basement": basement,
        "hotwaterheating": hotwaterheating,
        "airconditioning": airconditioning,
        "prefarea": prefarea,
        "furnishingstatus": furnishingstatus
    }

    input_df = pd.DataFrame([input_dict])

    # --- Apply transformations ---
    categorical_cols = input_df.select_dtypes(include="object").columns
    numerical_cols = input_df.select_dtypes(include=["int64", "float64"]).columns

    encoded = encoder.transform(input_df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    scaled = scaler.transform(input_df[numerical_cols])
    scaled_df = pd.DataFrame(scaled, columns=numerical_cols)

    final_input = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    # --- Prediction ---
    if st.button("Predict Price"):
        price = model.predict(final_input)[0]
        st.success(f"🏷️ Estimated House Price: ₹ {round(price):,}")
