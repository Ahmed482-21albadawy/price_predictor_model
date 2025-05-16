import sqlite3
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import json

# Load the saved model
model = joblib.load('price_prediction_model.pkl')

# Load the column order from the JSON file
with open('column_order.json', 'r') as f:
    column_order = json.load(f)
    

# List of all categories
categories = ['Dairy', 'Beverages', 'Snacks', 'Fruits', 'Vegetables']

# Initialize OneHotEncoder for categories
one_hot_encoder = OneHotEncoder(handle_unknown='ignore')
one_hot_encoder.fit([[category] for category in categories])  # fit the encoder on all categories


# Convert weight/volume to grams
def convert_to_grams(weight_volume):
    if weight_volume.endswith('L'):  # L => liters
        return float(weight_volume[:-1]) * 1000
    elif weight_volume.endswith('ml'):# ml => milliliters
        return float(weight_volume[:-2])
    elif weight_volume.endswith('kg'):   # kg => kilograms
        return float(weight_volume[:-2]) * 1000
    elif weight_volume.endswith('g'): # g => grams
        return float(weight_volume[:-1])
    else:
        raise ValueError(f"Unknown unit in weight_volume: {weight_volume}")

# predict the price
def predict_price(product_id):
    # Connect to database
    conn = sqlite3.connect('products.db')
    cursor = conn.cursor()

    # Fetch the data of the product
    cursor.execute('''
    SELECT ProductName, Category, Weight_Volume, Shelf_Life, Date, Price 
    FROM products
    WHERE ProductID = ?
    ''', (product_id,))

    # Get the data
    product_data = cursor.fetchone()
    print(type(product_data))
    print(product_data)

    if not product_data:
        return f"No data found for product ID: {product_id}"

    # Fetch all product names from the database to fit the encoder
    cursor.execute('SELECT ProductName FROM products')
    product_names = [row[0] for row in cursor.fetchall()]
    print(type(product_names))
    # Close the connection
    conn.close()

    label_encoder = LabelEncoder()
    label_encoder.fit(product_names)  # Fit the encoder on all product names

    # Prepare the features
    input_features = {
        'Product_Name_Encoded': label_encoder.transform([product_data[0]])[0],  # Encode product name
        'Weight_Volume_g': convert_to_grams(product_data[2]),  # Convert weight/volume to grams
        'Shelf_Life_Perishable': 1 if product_data[3] == 'Perishable' else 0,
        'Shelf_Life_Non-Perishable': 1 if product_data[3] == 'Non-Perishable' else 0,
        'Month': pd.to_datetime(product_data[4]).month,  # Extract month
        'Year': pd.to_datetime(product_data[4]).year,     # Extract year
        'Price': product_data[5]  # price
        
    }
    
    print(type(input_features))
    # Encode the category column using OneHotEncoder
    category_encoded = one_hot_encoder.transform([[product_data[1]]]).toarray().flatten()
    for i, category in enumerate(categories):
        input_features[f'Category_{category}'] = category_encoded[i]

    # Convert to DataFrame
    input_df = pd.DataFrame([input_features])
    
    # make the column order matches the training data order
    input_df = input_df[column_order]

    # predict
    predicted_price = model.predict(input_df)[0]
    return predicted_price

# Example
product_id = int(input("Enter the Product ID:"))  
predicted_price = predict_price(product_id)
print(f"Predicted price for product ID {product_id}: ${predicted_price:.2f}")
