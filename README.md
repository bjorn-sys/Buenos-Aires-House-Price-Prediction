# Buenos-Aires-House-Price-Prediction

predicting the house prices of apartments in buenos aires argentina

this project is about predicting the price of apartments in buenos aires argentina using linear regression model

i needed to clean the data and filter the data needed for this project using a wrangle function...seperating columns, creating new features from the existing one.

i used glob library to join all the csv files containing the real estate files needed for the data and then concatenated all of them into one dataframe

firstly i filtered the property type to only "APARTMENTS"

secondly i filtered the city to "CAPITAL FEDERAL"

the i filtered the price to less than "400,000.00" 

the surface area i worked with was from 31sq m2 to 101sq m2 afer removing the top and bottom 10%

i removed columns with high and low cardinality due to high or very low unique values in the features

i also removed columns with data leakage...columns that can give information to the target column before it might have predicted what is needed

dropped columns with more than 50% nan values

used simple imputer to fill the features needed for model development

after visualization i found out that there is a positive relationship between the surface area covered and the price of the apartment...the more area the more the price

also found the average price of the apartments to be within the region of 132k

visualizes the neighborhoods with the highest number of apartments is 'pelermo' neighborhood and avearge price in each neighborhood apartment are sold for and the highest is 'puerto madero'

mean absolute error baseline is  '44860.1' while mean absolute error training " 4330.4" which is lower than the baseline and shows good model performance.

created a function for inputing data for the model to predict and came out with result.

thanks
