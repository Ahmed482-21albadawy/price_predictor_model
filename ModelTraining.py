import pandas as pd

DS = pd.read_csv('ProductPricing.csv')


## filling the last missing value of each product in next month price column
# Loop through each unique Product ID
for product_id in DS['ProductID'].unique():
    # Filter the rows for the current product
    product_rows = DS[DS['ProductID'] == product_id]
    
    # Ensure the product has at least 24 records
    if len(product_rows) >= 24:
        # Calculate the mean of the first 23 'Next Month Price' values
        mean_next_month_price = product_rows['Next_Month_Price'].iloc[:23].mean()
        
        # Get the index of the last row for this product
        last_index = product_rows.index[-1]
        
        # Assign the mean value to the last row's 'Next Month Price'
        DS.at[last_index, 'Next_Month_Price'] = mean_next_month_price

# Check missing values
print(DS.isnull().sum())

#Remove missing values
DS = DS.dropna()

# Check duplicates
print(DS.duplicated().sum())

# Remove duplicates
DS = DS.drop_duplicates()

# print data types
print(DS.dtypes)

########   Convert DataTypes  ###########
# convert to datetime
DS['Date'] = pd.to_datetime(DS['Date'])

# Convert categorical columns to 'category' type
DS['ProductName'] = DS['ProductName'].astype('category')
DS['Category'] = DS['Category'].astype('category')
DS['Shelf_Life'] = DS['Shelf_Life'].astype('category')

# create month & year columns from "Date"
DS['Month'] = DS['Date'].dt.month
DS['Year'] = DS['Date'].dt.year

# One-hot encode categorical columns
DS = pd.get_dummies(DS, columns=['Category', 'Shelf_Life'])


# Function to convert weight/volume to grams
def convert_to_grams(weight_volume):
    if weight_volume.endswith('L'): # L => liters
        return float(weight_volume[:-1]) * 1000
    elif weight_volume.endswith('ml'): # ml => milliliters
        return float(weight_volume[:-2])
    elif weight_volume.endswith('kg'): # kg => kilograms
        return float(weight_volume[:-2]) * 1000
    elif weight_volume.endswith('g'): # g => grams
        return float(weight_volume[:-1])
    else:
        raise ValueError(f"Unknown unit in weight_volume: {weight_volume}")

#  Weight/Volume column's values to grams
DS['Weight_Volume_g'] = DS['Weight_Volume'].apply(convert_to_grams)

# Drop 'Weight/Volume' column
DS = DS.drop(columns=['Weight_Volume'])

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode the Product Name
DS['Product_Name_Encoded'] = label_encoder.fit_transform(DS['ProductName'])

# Drop the Product Name column
DS = DS.drop(columns=['ProductName'])

# Sort the data by Date column
DS = DS.sort_values(by='Date')

# Define the cutoff date
cutoff_date = '2023-12-01'

# Split to training and testing data
train = DS[DS['Date'] < cutoff_date]
test = DS[DS['Date'] >= cutoff_date]


# Splitting the data into X(train/test) & y(train/test)
X_train = train.drop(columns=['Next_Month_Price', 'Date', 'ProductID'])
X_test = test.drop(columns=['Next_Month_Price', 'Date', 'ProductID'])
y_train = train['Next_Month_Price']
y_test = test['Next_Month_Price']


# Check the shape of the training and testing data
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


import json

# Save the column order to a JSON file
with open('column_order.json', 'w') as f:
    json.dump(X_train.columns.tolist(), f)


######   Training   #######
from sklearn.neighbors import KNeighborsRegressor

# Initialize the model
model = KNeighborsRegressor(n_neighbors=5)

# Train
model.fit(X_train, y_train)


# Predict on the testing data
y_pred = model.predict(X_test)

from sklearn.metrics import r2_score
print("##########################")
# Calculate R2 metric
r2 = r2_score(y_test, y_pred)

print(f"Test_R² Score: {r2:.3f}")

# predict on the training data
y_train_pred = model.predict(X_train)

# Calculate R2 metric
train_r2 = r2_score(y_train, y_train_pred)

print(f"Train_R² Score: {train_r2:.3f}")
print("##########################")


# Save the trained model
import joblib
joblib.dump(model, 'price_prediction_model.pkl')
