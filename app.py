import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set page configuration
st.set_page_config(
    page_title="Buenos Aires Real Estate Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
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
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #d4edda;
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class RealEstatePredictor:
    def __init__(self):
        self.scaler = None
        self.model = None
        self.neighborhoods = None
        self.expected_features = None
        self.load_models()
    
    def load_models(self):
        try:
            with open('scaler.pkl', 'rb') as f:
                self.scaler = pickle.load(f)
            with open('best_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            
            # Get the actual feature names from the scaler
            if hasattr(self.scaler, 'feature_names_in_'):
                self.expected_features = list(self.scaler.feature_names_in_)
                st.success(f"Loaded scaler with {len(self.expected_features)} features")
            else:
                # Fallback: use the neighborhoods from your original data
                self.expected_features = self._get_fallback_features()
            
            # Extract neighborhood names from feature names
            self.neighborhoods = self._extract_neighborhoods()
            
        except FileNotFoundError:
            st.error("Model files not found. Please ensure 'scaler.pkl' and 'best_model.pkl' are in the same directory.")
    
    def _get_fallback_features(self):
        """Fallback feature list based on your original data"""
        base_features = ['surface_covered_in_m2', 'rooms', 'lat', 'lon']
        neighborhood_features = [f'neighborhood_{n}' for n in self._get_all_neighborhoods()]
        return base_features + neighborhood_features
    
    def _get_all_neighborhoods(self):
        """Get all neighborhoods including the empty one"""
        neighborhoods = [
            'Caballito', 'Constitución', 'Once', 'Almagro', 'Palermo', 'Flores', 'Belgrano',
            'Liniers', 'Villa Crespo', 'San Cristobal', 'Congreso', 'Saavedra', 'Balvanera',
            'Parque Avellaneda', 'San Telmo', 'Nuñez', 'Recoleta', 'Barrio Norte', 'Abasto',
            'Centro / Microcentro', 'Paternal', 'Chacarita', 'Mataderos', '', 'Coghlan', 'Las Cañitas',
            'Villa Urquiza', 'Monserrat', 'Villa Pueyrredón', 'San Nicolás', 'Villa del Parque',
            'Villa Luro', 'Parque Chacabuco', 'Boedo', 'Parque Centenario', 'Parque Chas',
            'Colegiales', 'Villa Ortuzar', 'Villa Devoto', 'Villa Lugano', 'Floresta', 'Barracas',
            'Retiro', 'Versalles', 'Boca', 'Puerto Madero', 'Agronomía', 'Monte Castro',
            'Tribunales', 'Parque Patricios', 'Velez Sarsfield', 'Villa General Mitre',
            'Villa Santa Rita', 'Villa Soldati', 'Villa Real', 'Pompeya'
        ]
        return neighborhoods
    
    def _extract_neighborhoods(self):
        """Extract neighborhood names from feature names"""
        neighborhoods = []
        for feature in self.expected_features:
            if feature.startswith('neighborhood_'):
                neighborhood_name = feature.replace('neighborhood_', '')
                neighborhoods.append(neighborhood_name)
        
        # Filter out empty string and sort
        neighborhoods = [n for n in neighborhoods if n]  # Remove empty strings
        return sorted(neighborhoods)
    
    def preprocess_input(self, surface_area, rooms, lat, lon, neighborhood):
        """Preprocess user input for prediction"""
        # Create a DataFrame with zeros for all expected features
        input_data = pd.DataFrame(np.zeros((1, len(self.expected_features))), 
                                columns=self.expected_features)
        
        # Set the basic features
        input_data['surface_covered_in_m2'] = surface_area
        input_data['rooms'] = rooms
        input_data['lat'] = lat
        input_data['lon'] = lon
        
        # Set the neighborhood (one-hot encoded)
        neighborhood_col = f'neighborhood_{neighborhood}'
        if neighborhood_col in input_data.columns:
            input_data[neighborhood_col] = 1
        else:
            st.warning(f"Neighborhood '{neighborhood}' not found in trained features. Using default encoding.")
        
        return input_data
    
    def predict_price(self, input_features):
        """Make prediction using the trained model"""
        if self.model and self.scaler:
            try:
                # Debug: show feature alignment
                st.write(f"🔍 Input features: {len(input_features.columns)}")
                st.write(f"🔍 Expected features: {len(self.expected_features)}")
                
                # Ensure the input features match exactly what the scaler expects
                missing_features = set(self.expected_features) - set(input_features.columns)
                if missing_features:
                    st.error(f"Missing features: {missing_features}")
                    return None
                
                # Reorder columns to match scaler expectation
                input_features = input_features[self.expected_features]
                
                # Scale the features
                scaled_features = self.scaler.transform(input_features)
                
                # Make prediction
                prediction = self.model.predict(scaled_features)[0]
                return max(0, prediction)  # Ensure non-negative price
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return None
        return None

def main():
    # Initialize predictor
    predictor = RealEstatePredictor()
    
    # Header
    st.markdown('<h1 class="main-header">🏠 Buenos Aires Real Estate Price Predictor</h1>', unsafe_allow_html=True)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", 
                                   ["📊 Data Overview", "🔮 Price Prediction", "📈 Model Performance"])
    
    if app_mode == "📊 Data Overview":
        show_data_overview(predictor)
    elif app_mode == "🔮 Price Prediction":
        show_price_prediction(predictor)
    elif app_mode == "📈 Model Performance":
        show_model_performance()

def show_data_overview(predictor):
    st.header("📊 Dataset Overview")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Key Statistics")
        stats_data = {
            'Metric': ['Total Apartments', 'Average Price (USD)', 'Average Surface Area (m²)', 
                      'Average Rooms', 'Number of Neighborhoods'],
            'Value': ['4,876', '$132,384', '53.66 m²', '2.31', '56']
        }
        stats_df = pd.DataFrame(stats_data)
        st.dataframe(stats_df, use_container_width=True)
        
        # Show available neighborhoods
        if predictor.neighborhoods:
            st.subheader("Available Neighborhoods")
            st.write(f"Total: {len(predictor.neighborhoods)} neighborhoods")
            neighborhoods_text = ", ".join(predictor.neighborhoods[:10]) + "..." if len(predictor.neighborhoods) > 10 else ", ".join(predictor.neighborhoods)
            st.write(neighborhoods_text)
    
    with col2:
        st.subheader("Price Distribution")
        # Simulate price distribution based on your actual data statistics
        np.random.seed(42)
        prices = np.random.normal(132384, 58744, 1000)
        prices = prices[(prices > 0) & (prices < 400000)]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(prices, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_xlabel('Price (USD)')
        ax.set_ylabel('Frequency')
        ax.set_title('Price Distribution')
        st.pyplot(fig)
    
    # Feature relationships
    st.subheader("Feature Relationships")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Surface Area vs Price
        st.write("**Surface Area vs Price**")
        fig, ax = plt.subplots(figsize=(10, 6))
        # Simulate data based on your actual relationships
        surface_area = np.random.uniform(30, 100, 200)
        price = surface_area * 2000 + np.random.normal(0, 20000, 200)
        ax.scatter(surface_area, price, alpha=0.6, color='blue')
        ax.set_xlabel('Surface Area (m²)')
        ax.set_ylabel('Price (USD)')
        ax.set_title('Price vs Surface Area')
        st.pyplot(fig)
    
    with col2:
        # Neighborhood prices
        st.write("**Top 10 Neighborhoods by Average Price**")
        neighborhood_data = {
            'Neighborhood': ['Puerto Madero', 'Recoleta', 'Palermo', 'Belgrano', 'Barrio Norte',
                           'Nuñez', 'Las Cañitas', 'Colegiales', 'Villa Urquiza', 'Caballito'],
            'Average Price (USD)': [280000, 220000, 190000, 175000, 170000, 
                                  165000, 160000, 155000, 150000, 145000]
        }
        neighborhood_df = pd.DataFrame(neighborhood_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(neighborhood_df['Neighborhood'], neighborhood_df['Average Price (USD)'], 
                color='lightcoral')
        ax.set_xlabel('Average Price (USD)')
        ax.set_title('Top 10 Neighborhoods by Average Price')
        plt.tight_layout()
        st.pyplot(fig)

def show_price_prediction(predictor):
    st.header("🔮 Price Prediction")
    
    st.write("""
    Enter the details of the apartment to get a price prediction. 
    The model predicts prices for apartments in Buenos Aires Capital Federal under $400,000 USD.
    """)
    
    # Check if model is loaded
    if predictor.model is None or predictor.scaler is None:
        st.error("❌ Model not loaded properly. Please check if model files are available.")
        return
    
    # Display model info
    st.info(f"✅ Model loaded: {len(predictor.expected_features)} features, {len(predictor.neighborhoods)} neighborhoods available")
    
    # Input form
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            surface_area = st.slider(
                "Surface Covered Area (m²)",
                min_value=30,
                max_value=100,
                value=50,
                help="Surface area in square meters"
            )
            
            rooms = st.slider(
                "Number of Rooms",
                min_value=1,
                max_value=6,
                value=2,
                help="Number of rooms in the apartment"
            )
            
            # Neighborhood selection
            if predictor.neighborhoods:
                neighborhood = st.selectbox(
                    "Neighborhood",
                    options=predictor.neighborhoods,
                    index=predictor.neighborhoods.index('Palermo') if 'Palermo' in predictor.neighborhoods else 0,
                    help="Select the neighborhood"
                )
            else:
                st.error("No neighborhoods loaded")
                return
        
        with col2:
            st.write("**Location Coordinates**")
            lat = st.number_input(
                "Latitude",
                min_value=-34.9,
                max_value=-34.5,
                value=-34.58,
                format="%.6f",
                help="Latitude coordinate (e.g., -34.58 for central Buenos Aires)"
            )
            
            lon = st.number_input(
                "Longitude",
                min_value=-58.6,
                max_value=-58.3,
                value=-58.43,
                format="%.6f",
                help="Longitude coordinate (e.g., -58.43 for central Buenos Aires)"
            )
            
            # Show coordinate help
            with st.expander("💡 Coordinate Help"):
                st.write("""
                **Typical Buenos Aires Coordinates:**
                - **Central Areas**: -34.58 to -34.62 latitude, -58.38 to -58.48 longitude
                - **Palermo**: -34.57 to -34.59
                - **Recoleta**: -34.58 to -34.59
                - **Downtown**: -34.60 to -34.61
                """)
        
        # Submit button
        submitted = st.form_submit_button("Predict Price", use_container_width=True)
    
    # Make prediction when form is submitted
    if submitted:
        # Show loading spinner
        with st.spinner('Making prediction...'):
            # Preprocess input
            input_features = predictor.preprocess_input(surface_area, rooms, lat, lon, neighborhood)
            
            # Make prediction
            predicted_price = predictor.predict_price(input_features)
        
        if predicted_price is not None:
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2>Predicted Price: ${predicted_price:,.2f} USD</h2>
                <p>Based on the features provided</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show feature insights
            st.subheader("📊 Feature Insights")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Surface Area",
                    f"{surface_area} m²",
                    f"${surface_area * 2000:,.0f} est. impact"
                )
            
            with col2:
                st.metric(
                    "Rooms",
                    f"{rooms}",
                    f"${rooms * 15000:,.0f} est. impact"
                )
            
            with col3:
                neighborhood_impact = 50000 if neighborhood in ['Puerto Madero', 'Recoleta', 'Palermo'] else 30000
                st.metric(
                    "Neighborhood",
                    neighborhood,
                    f"${neighborhood_impact:,.0f} est. impact"
                )
            
            # Price analysis
            st.subheader("💰 Price Analysis")
            
            analysis_col1, analysis_col2, analysis_col3 = st.columns(3)
            
            with analysis_col1:
                st.metric(
                    "Budget Range",
                    "$100K - $200K",
                    f"${predicted_price - 150000:,.0f} from mid-range",
                    delta_color="off"
                )
            
            with analysis_col2:
                avg_diff = predicted_price - 140000
                st.metric(
                    "Market Position",
                    "Average" if 120000 <= predicted_price <= 160000 else "Above Avg" if predicted_price > 160000 else "Below Avg",
                    f"${avg_diff:,.0f} from avg"
                )
            
            with analysis_col3:
                affordability = "High" if predicted_price < 100000 else "Medium" if predicted_price < 200000 else "Premium"
                st.metric(
                    "Affordability",
                    affordability,
                    delta=None
                )

def show_model_performance():
    st.header("📈 Model Performance Analysis")
    
    st.write("""
    This section shows the performance metrics of our trained machine learning models 
    for predicting real estate prices in Buenos Aires.
    """)
    
    # Model comparison metrics
    st.subheader("🏆 Model Performance Comparison")
    
    performance_data = {
        'Model': ['Random Forest (Tuned)', 'XGBoost (Tuned)', 'Linear Regression', 
                 'Decision Tree', 'KNN Regressor', 'SVR (Tuned)'],
        'MAE (USD)': [20869, 20880, 25849, 26460, 22285, 25887],
        'MSE': [930363931, 943924785, 1250891040, 1360222781, 1040248078, 1420596922],
        'R² Score': [0.7476, 0.7439, 0.6606, 0.6309, 0.7178, 0.6146]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    # Display metrics with formatting
    styled_df = perf_df.style.format({
        'MAE (USD)': '{:,.0f}',
        'MSE': '{:,.0f}',
        'R² Score': '{:.4f}'
    })
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📉 Mean Absolute Error (MAE) Comparison**")
        fig, ax = plt.subplots(figsize=(10, 6))
        models = perf_df['Model']
        mae = perf_df['MAE (USD)']
        
        colors = ['#FF6B6B' if 'Random Forest' in model else '#4ECDC4' for model in models]
        bars = ax.barh(models, mae, color=colors, alpha=0.8)
        ax.set_xlabel('MAE (USD) - Lower is Better')
        ax.set_title('Model Comparison: Mean Absolute Error')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 100, bar.get_y() + bar.get_height()/2, 
                   f'${width:,.0f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.write("**📈 R² Score Comparison**")
        fig, ax = plt.subplots(figsize=(10, 6))
        r2_scores = perf_df['R² Score']
        
        colors = ['#FF6B6B' if 'Random Forest' in model else '#4ECDC4' for model in models]
        bars = ax.barh(models, r2_scores, color=colors, alpha=0.8)
        ax.set_xlabel('R² Score - Higher is Better')
        ax.set_title('Model Comparison: R² Score')
        ax.set_xlim(0, 1)
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{width:.4f}', ha='left', va='center', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)

if __name__ == "__main__":
    main()