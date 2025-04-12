# Buenos-Aires-House-Price-Prediction

predicting the house prices of apartments in buenos aires argentina
this project is about predicting the price of apartments in buenos aires argentina using linear regression model
i needed to clean the data and filter the data needed for this project using a wrangle function...seperating columns, creating new features from the existing one.
i used glob library to join all the csv files containing the real estate files needed for te data and then concatenated all of them into one dataframe
firstly i needed to filter the property type to only "APARTMENTS"
secondly i needed to filter the city to "CAPITAL FEDERAL"
the i filtered the price to less than "400,000.00" 
the surface area i worked with was from 31sq m2 to 101sq m2
i removing columns with high and low cardinality due to high or very low unique values in the features
i also removed columns with data leakage...columns that can give information to the target column before it might have predicted what is needed
dropped columns with more than 50% nan values
used simple imputer to fill the features needed for model development
after visualization i found out that there is a positive relationship between the surface area covered and the price of the apartment...the more area the more the price
also found the average price of the apartments to be within the region of 132k
visualizes the neighborhoods with the highest number of apartments and avearge price in each neighborhood apartment are sold for
got the y_pred_training, mean absolute error baseline and mean absolute error training which is lower than the baseline and shows good model performance
created new data for the model to predict and came out with result.
thanks
