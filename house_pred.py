

import streamlit as st
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

# ---------------------
# Load and prepare data
# ---------------------
@st.cache_data
def load_data():
    import glob

    def wrangle(filepath):
        df = pd.read_csv(filepath)
        mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
        mask_apt = df["property_type"] == "apartment"
        mask_price = df["price_aprox_usd"] < 400_000
        df = df[mask_ba & mask_apt & mask_price]
        low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
        df = df[df["surface_covered_in_m2"].between(low, high)]
        df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
        df.drop(columns="lat-lon", inplace=True)
        df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
        df.drop(columns="place_with_parent_names", inplace=True)
        df.drop(columns=['expenses', 'floor', 'operation', 'currency', 'properati_url',
                         'price', 'price_aprox_local_currency', 'price_per_m2', 'price_usd_per_m2',
                         'surface_total_in_m2', 'rooms'], inplace=True)
        return df

    files = glob.glob(r'C:\Users\USER\Desktop\PROJECTS\buenos-aires-real-estate-*.csv')
    frames = [wrangle(file) for file in files]
    df = pd.concat(frames, ignore_index=True)
    df.drop(columns=["property_type"], inplace=True)
    return df

df = load_data()

# ---------------------
# Train the model
# ---------------------
features = ['surface_covered_in_m2', 'lat', 'lon', 'neighborhood']
X_train = df[features]
y_train = df['price_aprox_usd']

model = make_pipeline(
    OneHotEncoder(),
    SimpleImputer(),
    LinearRegression()
)
model.fit(X_train, y_train)

# ---------------------
# Streamlit Interface
# ---------------------
st.title("🏠 Buenos Aires Apartment Price Predictor")
st.write("Estimate apartment prices in Capital Federal based on area and location.")

# Input Widgets
area = st.number_input("Surface Area (sq meters)", min_value=30.0, max_value=120.0, value=50.0)
lat = st.number_input("Latitude", value=-34.60, format="%.5f")
lon = st.number_input("Longitude", value=-58.45, format="%.5f")
neighborhoods = sorted(df['neighborhood'].dropna().unique())
neighborhood = st.selectbox("Neighborhood", neighborhoods)

# Prediction
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'surface_covered_in_m2': [area],
        'lat': [lat],
        'lon': [lon],
        'neighborhood': [neighborhood]
    })

    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Apartment Price: ${prediction:,.2f}")




