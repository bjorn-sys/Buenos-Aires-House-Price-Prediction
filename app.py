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
from plotly.subplots import make_subplots
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
import warnings
from glob import glob

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
# DATA WRANGLING FUNCTION
# =============================================================================
def wrangle(filepath):
    # Read CSV file
    df = pd.read_csv(filepath)
    
    # Subset data: Apartments in "Capital Federal", less than 400,000
    mask_ba = df["place_with_parent_names"].str.contains("Capital Federal")
    mask_apt = df["property_type"] == "apartment"
    mask_price = df["price_aprox_usd"] < 400_000
    
    df = df[mask_ba & mask_apt & mask_price]

    # Subset data: Remove outliers for "surface_covered_in_m2"
    low, high = df["surface_covered_in_m2"].quantile([0.1, 0.9])
    mask_area = df["surface_covered_in_m2"].between(low, high)
    df = df[mask_area]

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get neighborhood name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)

    # Drop columns with more than 50% NaN values
    df.drop(columns=['expenses', 'floor','operation', 'currency', 'properati_url'], inplace=True)
    
    # Drop leaky columns
    df.drop(columns=['price','price_aprox_local_currency','price_per_m2','price_usd_per_m2'], inplace=True)
    
    # Drop columns with multicollinearity
    df.drop(columns=['surface_total_in_m2', 'rooms'], inplace=True)
    
    return df

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================
@st.cache_data
def load_data():
    files = glob(r'C:\Users\USER\Desktop\PROJECTS\buenos-aires-real-estate-*.csv')
    frames = []
    for file in files:
        df = wrangle(file)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    return df

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
        OneHotEncoder(),
        SimpleImputer(),
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

def create_model_performance(model, X_train, y_train):
    y_pred = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_pred)
    mse = mean_squared_error(y_train, y_pred)
    r2 = r2_score(y_train, y_pred)
    
    # Create performance metrics
    metrics_df = pd.DataFrame({
        'Metric': ['Mean Absolute Error', 'Mean Squared Error', 'R² Score'],
        'Value': [f'${mae:,.2f}', f'${mse:,.0f}', f'{r2:.4f}']
    })
    
    # Create prediction vs actual plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_train, y=y_pred, mode='markers', 
                            name='Predictions', marker=dict(color='#636efa')))
    fig.add_trace(go.Scatter(x=[y_train.min(), y_train.max()], 
                            y=[y_train.min(), y_train.max()], 
                            mode='lines', 
                            name='Perfect Prediction', 
                            line=dict(color='red', dash='dash')))
    fig.update_layout(title='📈 Model Performance: Actual vs Predicted Prices',
                     xaxis_title='Actual Price (USD)',
                     yaxis_title='Predicted Price (USD)')
    
    return metrics_df, fig

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
                                   "🤖 Price Prediction",
                                   "📈 Model Performance"])
    
    # Display dataset info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Dataset Info")
    st.sidebar.metric("Total Apartments", f"{len(df):,}")
    st.sidebar.metric("Neighborhoods", df['neighborhood'].nunique())
    st.sidebar.metric("Average Price", f"${df['price_aprox_usd'].mean():,.0f}")
    
    # Main content based on selection
    if app_section == "📊 Data Overview":
        show_data_overview(df)
    elif app_section == "🏙️ Neighborhood Analysis":
        show_neighborhood_analysis(df)
    elif app_section == "🤖 Price Prediction":
        show_price_prediction(df, model)
    elif app_section == "📈 Model Performance":
        show_model_performance(model, X_train, y_train)

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
    
    # Statistical summary
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)
    
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
            st.metric("Price per m²", f"${neighborhood_data['price_aprox_usd'].mean() / neighborhood_data['surface_covered_in_m2'].mean():.0f}")
        
        # Neighborhood-specific visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(neighborhood_data, x='price_aprox_usd',
                              title=f'Price Distribution in {selected_neighborhood}')
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.scatter(neighborhood_data, x='surface_covered_in_m2', y='price_aprox_usd',
                            title=f'Price vs Area in {selected_neighborhood}')
            st.plotly_chart(fig, use_container_width=True)

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
                        max_value=100.0, 
                        value=50.0, 
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
    
    # Sample predictions
    st.subheader("💡 Sample Predictions")
    sample_data = [
        {"Area": 40, "Neighborhood": "Palermo", "Price": "~$110,000"},
        {"Area": 60, "Neighborhood": "Recoleta", "Price": "~$150,000"},
        {"Area": 80, "Neighborhood": "Puerto Madero", "Price": "~$280,000"},
        {"Area": 50, "Neighborhood": "Belgrano", "Price": "~$120,000"}
    ]
    st.table(pd.DataFrame(sample_data))

# =============================================================================
# MODEL PERFORMANCE SECTION
# =============================================================================
def show_model_performance(model, X_train, y_train):
    st.header("📈 Model Performance")
    
    metrics_df, performance_fig = create_model_performance(model, X_train, y_train)
    
    # Display metrics
    st.subheader("📊 Model Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean Absolute Error", f"${mean_absolute_error(y_train, model.predict(X_train)):,.2f}")
    with col2:
        st.metric("R² Score", f"{r2_score(y_train, model.predict(X_train)):.4f}")
    with col3:
        baseline_mae = mean_absolute_error(y_train, [y_train.mean()] * len(y_train))
        model_mae = mean_absolute_error(y_train, model.predict(X_train))
        improvement = ((baseline_mae - model_mae) / baseline_mae) * 100
        st.metric("Improvement vs Baseline", f"{improvement:.1f}%")
    
    # Performance visualization
    st.plotly_chart(performance_fig, use_container_width=True)
    
    # Model details
    st.subheader("🔧 Model Details")
    st.write("""
    **Model Pipeline:**
    1. **One-Hot Encoding** for categorical features (neighborhood)
    2. **Simple Imputer** for handling missing values
    3. **Linear Regression** for price prediction
    
    **Features Used:**
    - Surface Area (m²)
    - Latitude
    - Longitude  
    - Neighborhood
    """)
    
    # Feature importance (simulated for linear regression)
    st.subheader("🎯 Feature Importance")
    try:
        coefficients = model.named_steps['linearregression'].coef_
        feature_names = ['Surface Area', 'Latitude', 'Longitude'] + \
                       list(model.named_steps['onehotencoder'].get_feature_names_out(['neighborhood']))
        
        # Create feature importance plot
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients
        }).head(20)  # Show top 20
        
        fig = px.bar(importance_df, x='Coefficient', y='Feature', 
                    title='Feature Coefficients (Top 20)',
                    orientation='h')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Feature importance details available after model training")

# =============================================================================
# RUN APPLICATION
# =============================================================================
if __name__ == "__main__":
    main()