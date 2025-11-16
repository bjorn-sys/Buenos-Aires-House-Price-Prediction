# 🏙️ Buenos Aires Apartment Price Predictor – Streamlit App
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pickle
from glob import glob
import warnings

# Machine Learning libraries
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

warnings.simplefilter(action="ignore", category=FutureWarning)

# Set page configuration
st.set_page_config(
    page_title="Buenos Aires Apartment Price Predictor",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------------------
# WRANGLE FUNCTION
# --------------------------------------------------------------

def wrangle(filepath):
    # Read CSV file
    if hasattr(filepath, 'read'):
        df = pd.read_csv(filepath)
    else:
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

    # Split "lat-lon" column
    df[["lat", "lon"]] = df["lat-lon"].str.split(",", expand=True).astype(float)
    df.drop(columns="lat-lon", inplace=True)

    # Get place name
    df["neighborhood"] = df["place_with_parent_names"].str.split("|", expand=True)[3]
    df.drop(columns="place_with_parent_names", inplace=True)
 
    # dropping columns with more than 50% NAN files
    df.drop(columns=['expenses', 'floor','operation', 'currency', 'properati_url'], inplace=True)
    
    # drop leaky columns
    df.drop(columns=['price','price_aprox_local_currency','price_per_m2','price_usd_per_m2'], inplace=True)
    
    # drop columns with multy collinearity
    df.drop(columns=['surface_total_in_m2'], inplace=True)
    
    return df

# --------------------------------------------------------------
# SIDEBAR - DATA UPLOAD
# --------------------------------------------------------------

st.sidebar.title("📁 Data Configuration")

# File upload option
use_sample_data = st.sidebar.checkbox("Use sample data from local directory", value=True)

if use_sample_data:
    files = glob('buenos-aires-real-estate-*.csv')
    if not files:
        files = glob('*.csv')
    if files:
        st.sidebar.success(f"Found {len(files)} CSV file(s)")
    else:
        st.sidebar.warning("No CSV files found in directory")
        files = []
else:
    uploaded_files = st.sidebar.file_uploader(
        "Upload CSV files", 
        type=["csv"], 
        accept_multiple_files=True,
        help="Upload Buenos Aires real estate CSV files"
    )
    files = uploaded_files if uploaded_files else []

# --------------------------------------------------------------
# MAIN APP
# --------------------------------------------------------------

# Header
st.markdown('<h1 class="main-header">🏙️ Buenos Aires Apartment Price Predictor</h1>', unsafe_allow_html=True)
st.markdown("---")

# Data Processing Section
if not files:
    st.info("👆 Please upload CSV files or use sample data from the sidebar to get started.")
    st.stop()

# Process data
with st.spinner("🔄 Processing and cleaning data..."):
    try:
        frames = []
        for file in files:
            df_temp = wrangle(file)
            frames.append(df_temp)
        
        if not frames:
            st.error("❌ No valid data found after processing. Please check your files.")
            st.stop()
            
        df = pd.concat(frames, ignore_index=True)
        
        # Clean data
        initial_shape = df.shape
        df.dropna(inplace=True)
        df.drop_duplicates(inplace=True)
        final_shape = df.shape
        
        st.session_state.df = df
        st.session_state.data_loaded = True
        
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        st.stop()

# Display data overview
if st.session_state.data_loaded:
    df = st.session_state.df
    
    # Data Overview Cards
    st.subheader("📊 Dataset Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Apartments", f"{len(df):,}")
    with col2:
        st.metric("Average Price", f"${df['price_aprox_usd'].mean():,.0f}")
    with col3:
        st.metric("Average Area", f"{df['surface_covered_in_m2'].mean():.0f} m²")
    with col4:
        st.metric("Average Rooms", f"{df['rooms'].mean():.1f}")
    with col5:
        st.metric("Neighborhoods", f"{df['neighborhood'].nunique()}")
    
    st.markdown("---")
    
    # --------------------------------------------------------------
    # EXPLORATORY DATA ANALYSIS
    # --------------------------------------------------------------
    
    st.header("📈 Exploratory Data Analysis")
    
    # Tabs for different visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏘️ Neighborhood Analysis", 
        "📊 Price Distribution", 
        "🗺️ Location Map", 
        "📐 Area vs Price", 
        "🔍 Raw Data"
    ])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Top 20 neighborhoods by average price
            st.subheader("🏆 Top 20 Neighborhoods by Average Price")
            top_price_neighborhoods = df.groupby('neighborhood')['price_aprox_usd'].mean().sort_values(ascending=False).head(20)
            
            fig = px.bar(
                top_price_neighborhoods,
                orientation='h',
                color=top_price_neighborhoods.values,
                color_continuous_scale='viridis',
                title='Average Price by Neighborhood (Top 20)'
            )
            fig.update_layout(yaxis_title='Neighborhood', xaxis_title='Average Price (USD)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top 20 neighborhoods by apartment count
            st.subheader("🏢 Top 20 Neighborhoods by Apartment Count")
            top_count_neighborhoods = df['neighborhood'].value_counts().head(20)
            
            fig = px.bar(
                top_count_neighborhoods,
                orientation='h',
                color=top_count_neighborhoods.values,
                color_continuous_scale='plasma',
                title='Number of Apartments by Neighborhood (Top 20)'
            )
            fig.update_layout(yaxis_title='Neighborhood', xaxis_title='Number of Apartments')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Price Distribution")
            fig = px.histogram(
                df, 
                x='price_aprox_usd',
                nbins=30,
                color_discrete_sequence=['#FF4B4B'],
                title='Distribution of Apartment Prices'
            )
            fig.update_layout(xaxis_title='Price (USD)', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🛏️ Rooms Distribution")
            fig = px.histogram(
                df,
                x='rooms',
                color_discrete_sequence=['#00CC96'],
                title='Distribution of Number of Rooms'
            )
            fig.update_layout(xaxis_title='Number of Rooms', yaxis_title='Count')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🗺️ Apartment Locations in Buenos Aires")
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            color='price_aprox_usd',
            size='surface_covered_in_m2',
            hover_data=['rooms', 'neighborhood'],
            color_continuous_scale='viridis',
            zoom=10,
            height=600,
            title='Apartment Locations Colored by Price'
        )
        fig.update_layout(mapbox_style="open-street-map")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("📐 Price vs Surface Area")
        fig = px.scatter(
            df,
            x='surface_covered_in_m2',
            y='price_aprox_usd',
            color='rooms',
            size='rooms',
            hover_data=['neighborhood'],
            title='Relationship between Surface Area and Price',
            color_continuous_scale='viridis'
        )
        fig.update_layout(xaxis_title='Surface Area (m²)', yaxis_title='Price (USD)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.subheader("🔍 Raw Data Preview")
        st.dataframe(df, use_container_width=True)
        
        # Download cleaned data
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Data",
            data=csv,
            file_name="buenos_aires_cleaned_apartments.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    
    # --------------------------------------------------------------
    # MODEL TRAINING SECTION
    # --------------------------------------------------------------
    
    st.header("🤖 Machine Learning Models")
    
    # Prepare data for modeling
    with st.spinner("🔄 Preparing data for machine learning..."):
        # Drop property_type and one-hot encode neighborhood
        df_model = df.drop(columns=['property_type'], errors='ignore')
        
        # One-hot encode neighborhood
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoded_neighborhood = encoder.fit_transform(df_model[['neighborhood']])
        encoded_cols = encoder.get_feature_names_out(['neighborhood'])
        
        df_encoded = pd.concat([
            df_model.drop('neighborhood', axis=1),
            pd.DataFrame(encoded_neighborhood, columns=encoded_cols, index=df_model.index)
        ], axis=1)
        
        # Split features and target
        X = df_encoded.drop(columns=['price_aprox_usd'])
        y = df_encoded['price_aprox_usd']
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        st.session_state.X_train_scaled = X_train_scaled
        st.session_state.X_test_scaled = X_test_scaled
        st.session_state.encoder = encoder
        st.session_state.scaler = scaler
    
    # Model training
    if st.button("🚀 Train All Models", type="primary"):
        with st.spinner("Training multiple machine learning models... This may take a few minutes."):
            models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0, solver='auto', random_state=42),
                'Decision Tree': DecisionTreeRegressor(max_depth=5, min_samples_split=10, min_samples_leaf=5, random_state=42),
                'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=7, min_samples_split=10,
                                                    min_samples_leaf=5, random_state=42, n_jobs=-1),
                'XGBoost': XGBRegressor(n_estimators=200, max_depth=7, learning_rate=0.1, subsample=0.8,
                                    colsample_bytree=0.8, random_state=42, n_jobs=-1),
                'KNN Regressor': KNeighborsRegressor(n_neighbors=5, weights='distance', metric='minkowski'),
                'SVR': SVR(C=100, epsilon=0.1, kernel='rbf', gamma='scale')
            }
            
            results = []
            best_model = None
            best_r2 = -np.inf
            
            progress_bar = st.progress(0)
            for i, (name, model) in enumerate(models.items()):
                # Update progress
                progress = (i + 1) / len(models)
                progress_bar.progress(progress)
                
                # Train model
                model.fit(st.session_state.X_train_scaled, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test_scaled)
                
                # Calculate metrics
                mae = mean_absolute_error(st.session_state.y_test, y_pred)
                mse = mean_squared_error(st.session_state.y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(st.session_state.y_test, y_pred)
                
                results.append({
                    'Model': name,
                    'MAE': mae,
                    'MSE': mse,
                    'RMSE': rmse,
                    'R²': r2
                })
                
                # Track best model
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
            
            # Display results
            results_df = pd.DataFrame(results).round(2)
            results_df = results_df.sort_values('R²', ascending=False)
            
            st.subheader("📊 Model Performance Comparison")
            st.dataframe(results_df, use_container_width=True)
            
            # Highlight best model
            best_model_name = results_df.iloc[0]['Model']
            st.success(f"🎯 Best Performing Model: **{best_model_name}** (R²: {results_df.iloc[0]['R²']:.4f})")
            
            # Store best model
            st.session_state.best_model = best_model
            st.session_state.models_trained = True
            st.session_state.results_df = results_df
            
            # Save model and scaler
            with open('scaler.pkl', 'wb') as f:
                pickle.dump(scaler, f)
            with open('rf_model.pkl', 'wb') as f:
                pickle.dump(best_model, f)
            
            st.balloons()
    
    # --------------------------------------------------------------
    # PRICE PREDICTION INTERFACE
    # --------------------------------------------------------------
    
    st.markdown("---")
    st.header("🔮 Price Prediction")
    
    if 'models_trained' in st.session_state and st.session_state.models_trained:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏠 Apartment Details")
            surface_area = st.slider("Surface Area (m²)", 30, 150, 70, help="Total covered area in square meters")
            rooms = st.selectbox("Number of Rooms", [1, 2, 3, 4, 5, 6], help="Number of rooms in the apartment")
            
            # Use actual data ranges for location
            lat_min, lat_max = df['lat'].min(), df['lat'].max()
            lon_min, lon_max = df['lon'].min(), df['lon'].max()
            
            latitude = st.slider("Latitude", float(lat_min), float(lat_max), float((lat_min + lat_max) / 2), 0.001,
                               help="Geographical latitude coordinate")
            longitude = st.slider("Longitude", float(lon_min), float(lon_max), float((lon_min + lon_max) / 2), 0.001,
                                help="Geographical longitude coordinate")
        
        with col2:
            st.subheader("📍 Neighborhood")
            neighborhoods = st.session_state.encoder.categories_[0]
            neighborhood = st.selectbox("Select Neighborhood", neighborhoods, 
                                      help="Choose the neighborhood where the apartment is located")
            
            if st.button("🎯 Predict Price", type="primary", use_container_width=True):
                try:
                    # Create one-hot vector for neighborhood
                    neigh_vector = np.zeros(len(st.session_state.encoder.get_feature_names_out(['neighborhood'])))
                    idx = list(st.session_state.encoder.get_feature_names_out(['neighborhood'])).index(f"neighborhood_{neighborhood}")
                    neigh_vector[idx] = 1
                    
                    # Prepare input data
                    input_data = np.array([[surface_area, rooms, latitude, longitude] + list(neigh_vector)])
                    input_scaled = st.session_state.scaler.transform(input_data)
                    
                    # Make prediction
                    pred = st.session_state.best_model.predict(input_scaled)[0]
                    
                    # Display prediction
                    st.markdown("---")
                    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
                    st.metric("Predicted Price", f"${pred:,.0f} USD")
                    st.metric("Price per m²", f"${pred/surface_area:,.0f} USD/m²")
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Show confidence interval (using MAE from best model)
                    best_model_results = st.session_state.results_df[st.session_state.results_df['Model'] == 'XGBoost'].iloc[0]
                    mae = best_model_results['MAE']
                    
                    st.info(f"📊 Expected price range: **${pred-mae:,.0f} - ${pred+mae:,.0f} USD**")
                    
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
    else:
        st.info("👆 Please train the models first using the 'Train All Models' button above.")
    
    # --------------------------------------------------------------
    # MODEL INSIGHTS
    # --------------------------------------------------------------
    
    if 'models_trained' in st.session_state and st.session_state.models_trained:
        st.markdown("---")
        st.header("🔍 Model Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Model Performance Comparison")
            fig = px.bar(
                st.session_state.results_df,
                x='Model',
                y='R²',
                color='R²',
                color_continuous_scale='viridis',
                title='R² Score by Model (Higher is Better)'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Error Metrics Comparison")
            fig = px.bar(
                st.session_state.results_df,
                x='Model',
                y='MAE',
                color='MAE',
                color_continuous_scale='reds',
                title='Mean Absolute Error by Model (Lower is Better)'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
    
    # --------------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------------
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🏙️ Buenos Aires Apartment Price Predictor | Built with Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

else:
    st.warning("Please load data to continue.")