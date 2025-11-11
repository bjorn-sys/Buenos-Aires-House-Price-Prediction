# --------------------------------------------------------------------
# 🏠 Buenos Aires Real Estate Price Predictor - Streamlit App
# --------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import warnings
import os

warnings.simplefilter(action="ignore", category=FutureWarning)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Buenos Aires Real Estate",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CUSTOM STYLING
# =============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 800;
    }
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e1e8ed;
    }
    .prediction-card {
        background: linear-gradient(135deg, #51cf66 0%, #40c057 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        font-weight: bold;
        font-size: 1.5rem;
        box-shadow: 0 8px 15px rgba(81, 207, 102, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SAMPLE DATA GENERATION (for demo when real data isn't available)
# =============================================================================
def generate_sample_data():
    """Generate realistic sample data for demonstration"""
    np.random.seed(42)
    n_samples = 2000
    
    # Neighborhoods in Buenos Aires
    neighborhoods = [
        'Palermo', 'Recoleta', 'Belgrano', 'Puerto Madero', 'Caballito',
        'Almagro', 'Flores', 'Constitución', 'Balvanera', 'San Telmo',
        'Villa Crespo', 'Chacarita', 'Colegiales', 'Nuñez', 'Saavedra'
    ]
    
    # Generate sample data
    data = {
        'property_type': ['apartment'] * n_samples,
        'price_aprox_usd': np.random.normal(150000, 50000, n_samples).clip(50000, 400000),
        'surface_covered_in_m2': np.random.normal(60, 20, n_samples).clip(30, 120),
        'lat': np.random.normal(-34.60, 0.05, n_samples),
        'lon': np.random.normal(-58.45, 0.05, n_samples),
        'neighborhood': np.random.choice(neighborhoods, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Adjust prices based on neighborhood (make it realistic)
    price_multipliers = {
        'Puerto Madero': 2.0, 'Recoleta': 1.8, 'Palermo': 1.6, 'Belgrano': 1.4,
        'Nuñez': 1.3, 'Colegiales': 1.3, 'Villa Crespo': 1.2, 'Caballito': 1.1,
        'Almagro': 1.0, 'Chacarita': 0.9, 'Balvanera': 0.9, 'San Telmo': 0.9,
        'Constitución': 0.8, 'Flores': 0.8, 'Saavedra': 0.8
    }
    
    for neighborhood, multiplier in price_multipliers.items():
        mask = df['neighborhood'] == neighborhood
        df.loc[mask, 'price_aprox_usd'] = df.loc[mask, 'price_aprox_usd'] * multiplier
    
    # Adjust prices based on surface area
    df['price_aprox_usd'] = df['price_aprox_usd'] * (df['surface_covered_in_m2'] / 60)
    
    return df

# =============================================================================
# DATA WRANGLING FUNCTION
# =============================================================================
def wrangle(filepath):
    """Wrangle data from CSV file - with error handling for missing files"""
    try:
        # Read CSV file
        df = pd.read_csv(filepath)
        
        # Subset data: Apartments in "Capital Federal", less than 400,000
        mask_ba = df["place_with_parent_names"].str.contains("Capital Federal", na=False)
        mask_apt = df["property_type"] == "apartment"
        mask_price = df["price_aprox_usd"] < 400_000
        
        df = df[mask_ba & mask_apt & mask_price]

        # Subset data: Remove outliers for "surface_covered_in_m2"
        low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
        mask_area = df["surface_covered_in_m2"].between(low, high)
        df = df[mask_area]

        # Split "lat-lon" column if it exists
        if "lat-lon" in df.columns:
            df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
            df.drop(columns="lat-lon", inplace=True)

        # Get neighborhood name
        if "place_with_parent_names" in df.columns:
            df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
            df.drop(columns="place_with_parent_names", inplace=True)

        # Drop columns that might not exist
        columns_to_drop = ['expenses', 'floor', 'operation', 'currency', 'properati_url',
                          'price', 'price_aprox_local_currency', 'price_per_m2', 'price_usd_per_m2',
                          'surface_total_in_m2', 'rooms']
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df.drop(columns=existing_columns_to_drop, inplace=True)
        
        return df
    except Exception as e:
        st.warning(f"Could not load data from {filepath}: {e}")
        return pd.DataFrame()  # Return empty DataFrame

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
@st.cache_data
def load_data():
    """Load data with fallback to sample data"""
    try:
        # Try to load from CSV files
        import glob
        files = glob.glob('*.csv')  # Look for CSV files in current directory
        
        if not files:
            # If no CSV files found, try the original path (might work locally)
            files = glob.glob(r'C:\Users\USER\Desktop\PROJECTS\buenos-aires-real-estate-*.csv')
        
        frames = []
        for file in files:
            df = wrangle(file)
            if not df.empty:
                frames.append(df)
        
        if frames:
            df = pd.concat(frames, ignore_index=True)
            st.success(f"✅ Loaded {len(df)} real estate listings from {len(files)} files")
            return df
        else:
            # If no data loaded, generate sample data
            st.info("📊 Using sample data for demonstration. Upload your own CSV files for real analysis.")
            return generate_sample_data()
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("📊 Generating sample data for demonstration...")
        return generate_sample_data()

# =============================================================================
# BUILD MODEL
# =============================================================================
@st.cache_resource
def build_model(df):
    features = ['surface_covered_in_m2', 'lat', 'lon', 'neighborhood']
    target = 'price_aprox_usd'
    
    X_train = df[features]
    y_train = df[target]
    
    model = make_pipeline(
        OneHotEncoder(handle_unknown='ignore'),
        SimpleImputer(strategy='mean'),
        LinearRegression()
    )
    model.fit(X_train, y_train)
    
    return model, X_train, y_train

# =============================================================================
# PREDICTION FUNCTION
# =============================================================================
def make_prediction(model, area, lat, lon, neighborhood):
    data = {
        'surface_covered_in_m2': area,
        'lat': lat,
        'lon': lon,
        'neighborhood': neighborhood
    }
    df = pd.DataFrame(data, index=[0])
    prediction = model.predict(df)
    return prediction[0]

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================
def create_price_distribution(df):
    fig = px.histogram(df, x='price_aprox_usd', 
                      title='📊 Distribution of Apartment Prices',
                      labels={'price_aprox_usd': 'Price (USD)', 'count': 'Number of Apartments'},
                      color_discrete_sequence=['#636efa'])
    fig.update_layout(showlegend=False)
    return fig

def create_area_vs_price_scatter(df):
    fig = px.scatter(df, x='surface_covered_in_m2', y='price_aprox_usd',
                    title='🏠 Price vs Surface Area',
                    labels={'surface_covered_in_m2': 'Surface Area (m²)', 
                           'price_aprox_usd': 'Price (USD)'},
                    trendline='lowess',
                    color_discrete_sequence=['#00cc96'])
    return fig

def create_neighborhood_price_chart(df):
    top_20 = df.groupby('neighborhood')['price_aprox_usd'].mean().sort_values(ascending=False).head(20)
    fig = px.bar(x=top_20.index, y=top_20.values,
                title='🏙️ Top 20 Neighborhoods by Average Price',
                labels={'x': 'Neighborhood', 'y': 'Average Price (USD)'},
                color=top_20.values,
                color_continuous_scale='viridis')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_neighborhood_count_chart(df):
    top_20_count = df['neighborhood'].value_counts().head(20)
    fig = px.bar(x=top_20_count.index, y=top_20_count.values,
                title='📈 Top 20 Neighborhoods by Number of Apartments',
                labels={'x': 'Neighborhood', 'y': 'Number of Apartments'},
                color=top_20_count.values,
                color_continuous_scale='plasma')
    fig.update_layout(xaxis_tickangle=-45)
    return fig

def create_geo_map(df):
    fig = px.scatter_mapbox(df, 
                           lat='lat', 
                           lon='lon',
                           color='price_aprox_usd',
                           size='surface_covered_in_m2',
                           hover_data=['neighborhood', 'surface_covered_in_m2'],
                           color_continuous_scale='viridis',
                           zoom=10,
                           title='🗺️ Apartment Prices Across Buenos Aires')
    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":30,"l":0,"b":0})
    return fig

# =============================================================================
# MAIN APPLICATION
# =============================================================================
def main():
    # Header
    st.markdown('<div class="main-header">🏠 Buenos Aires Real Estate Price Predictor</div>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🔄 Loading data and building model...'):
        df = load_data()
        model, X_train, y_train = build_model(df)
    
    # Sidebar
    st.sidebar.title("🔍 Navigation")
    app_section = st.sidebar.radio("Go to", 
                                  ["📊 Data Overview", 
                                   "🏙️ Neighborhood Analysis", 
                                   "🤖 Price Prediction"])
    
    # Display dataset info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Dataset Info")
    st.sidebar.metric("Total Apartments", f"{len(df):,}")
    st.sidebar.metric("Neighborhoods", df['neighborhood'].nunique())
    st.sidebar.metric("Average Price", f"${df['price_aprox_usd'].mean():,.0f}")
    
    # File uploader for custom data
    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            custom_df = pd.read_csv(uploaded_file)
            st.sidebar.success(f"Uploaded {len(custom_df)} rows")
            # Here you could add logic to process the uploaded file
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")
    
    # Main content based on selection
    if app_section == "📊 Data Overview":
        show_data_overview(df)
    elif app_section == "🏙️ Neighborhood Analysis":
        show_neighborhood_analysis(df)
    elif app_section == "🤖 Price Prediction":
        show_price_prediction(df, model)

# =============================================================================
# DATA OVERVIEW SECTION
# =============================================================================
def show_data_overview(df):
    st.header("📊 Dataset Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Average Price", f"${df['price_aprox_usd'].mean():,.0f}")
    with col2:
        st.metric("Average Area", f"{df['surface_covered_in_m2'].mean():.1f} m²")
    with col3:
        st.metric("Min Price", f"${df['price_aprox_usd'].min():,.0f}")
    with col4:
        st.metric("Max Price", f"${df['price_aprox_usd'].max():,.0f}")
    
    # Data preview
    st.subheader("📋 Data Sample")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Visualizations
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_price_distribution(df), use_container_width=True)
    with col2:
        st.plotly_chart(create_area_vs_price_scatter(df), use_container_width=True)
    
    # Geographical map
    st.subheader("🗺️ Geographical Distribution")
    st.plotly_chart(create_geo_map(df), use_container_width=True)

# =============================================================================
# NEIGHBORHOOD ANALYSIS SECTION
# =============================================================================
def show_neighborhood_analysis(df):
    st.header("🏙️ Neighborhood Analysis")
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(create_neighborhood_price_chart(df), use_container_width=True)
    with col2:
        st.plotly_chart(create_neighborhood_count_chart(df), use_container_width=True)
    
    # Interactive neighborhood selector
    st.subheader("🔍 Explore Specific Neighborhood")
    neighborhoods = sorted(df['neighborhood'].unique())
    selected_neighborhood = st.selectbox("Choose a neighborhood:", neighborhoods)
    
    if selected_neighborhood:
        neighborhood_data = df[df['neighborhood'] == selected_neighborhood]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Price", f"${neighborhood_data['price_aprox_usd'].mean():,.0f}")
        with col2:
            st.metric("Average Area", f"{neighborhood_data['surface_covered_in_m2'].mean():.1f} m²")
        with col3:
            st.metric("Number of Listings", len(neighborhood_data))
        with col4:
            price_per_m2 = neighborhood_data['price_aprox_usd'].mean() / neighborhood_data['surface_covered_in_m2'].mean()
            st.metric("Price per m²", f"${price_per_m2:,.0f}")

# =============================================================================
# PRICE PREDICTION SECTION
# =============================================================================
def show_price_prediction(df, model):
    st.header("🤖 Apartment Price Prediction")
    
    st.info("""
    **Predict apartment prices in Buenos Aires based on:**
    - Surface area
    - Location coordinates
    - Neighborhood
    """)
    
    # Prediction form
    col1, col2 = st.columns(2)
    
    with col1:
        area = st.slider("🏠 Surface Area (m²)", 
                        min_value=30.0, 
                        max_value=120.0, 
                        value=60.0, 
                        step=1.0)
        
        neighborhoods = sorted(df['neighborhood'].unique())
        neighborhood = st.selectbox("🏙️ Neighborhood", neighborhoods)
    
    with col2:
        # Get approximate coordinates for selected neighborhood
        if neighborhood:
            neighborhood_data = df[df['neighborhood'] == neighborhood]
            avg_lat = neighborhood_data['lat'].mean()
            avg_lon = neighborhood_data['lon'].mean()
        else:
            avg_lat = -34.60
            avg_lon = -58.46
        
        lat = st.number_input("📍 Latitude", 
                             value=float(avg_lat), 
                             format="%.6f")
        lon = st.number_input("📍 Longitude", 
                             value=float(avg_lon), 
                             format="%.6f")
    
    # Prediction button
    if st.button("🎯 Predict Price", type="primary", use_container_width=True):
        with st.spinner('Calculating prediction...'):
            prediction = make_prediction(model, area, lat, lon, neighborhood)
            
            # Display prediction
            st.markdown(f"""
            <div class="prediction-card">
                💰 Predicted Price: ${prediction:,.0f} USD
            </div>
            """, unsafe_allow_html=True)
            
            # Additional insights
            col1, col2, col3 = st.columns(3)
            with col1:
                price_per_m2 = prediction / area
                st.metric("Price per m²", f"${price_per_m2:,.0f}")
            with col2:
                neighborhood_avg = df[df['neighborhood'] == neighborhood]['price_aprox_usd'].mean()
                diff = prediction - neighborhood_avg
                st.metric("vs Neighborhood Avg", f"${diff:,.0f}")
            with col3:
                st.metric("Surface Area", f"{area} m²")

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()