# Buenos Aires House Price Prediction

This project focuses on predicting apartment prices in Buenos Aires, Argentina, using a Linear Regression model. The primary goal is to develop a data-driven approach for estimating apartment prices based on several key features.

---

# Libraries

* pandas : for data manipulation

* Sklearn : for model development

* Matplotlib : for visualaization

* Mean-Absolute-Error : for evaluation

* Glob : for joining multiple datasets
  
---

# 📊 Project Overview

* Location: Buenos Aires, Argentina

* Property Type: Apartments only

* Objective: Predict apartment prices using machine learning (Linear Regression)



---

# 🧹 Data Wrangling & Preprocessing

# ✅ Data Collection

* Used the glob library to read and combine multiple .csv files containing real estate listings.

* Concatenated all datasets into a single DataFrame for analysis.


# ✅ Data Filtering

* Property Type: Filtered for APARTMENTS only.

* Location: Limited to the Capital Federal region.

* Price: Filtered apartments priced under $400,000.

*  Area: Retained apartments with a surface area between 31 m² and 101 m², removing the top and bottom 10% of the range to eliminate outliers.


# ✅ Data Cleaning

* Dropped columns:

* With high or low cardinality (features with too many or too few unique values).

* That led to data leakage (columns that revealed target information).

* With more than 50% missing values.

* Used Simple Imputer to fill in missing values necessary for model training.



---

# 📈 Exploratory Data Analysis

* Found a positive correlation between surface area and apartment price — larger apartments tend to cost more.

* The average apartment price was approximately $132,000.

# Identified:

* Palermo as the neighborhood with the highest number of listings.

* Puerto Madero as the most expensive neighborhood on average.




---

# 🤖 Model Development

* Used a Linear Regression model for price prediction.

* Baseline Mean Absolute Error (MAE): 44,860.1

* Model MAE (on training data): 4,330.4

* The significant lower MAE compared to the baseline indicates strong model performance.




---

# 🛠️ Model Deployment

* Built a custom prediction function to input new apartment data and return price predictions.



---

* ✅ Conclusion

This project successfully demonstrates how data preprocessing, feature engineering, and machine learning can be combined to build a reliable price prediction model for real estate in Buenos Aires. The model provides insights into key factors influencing apartment prices and offers a foundation for further development or deployment.


---

# 🙏 Acknowledgments

Thank you for reviewing this project!
