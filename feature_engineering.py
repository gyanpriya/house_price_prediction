import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def load_data(path=r'C:\Users\hp\Desktop\House_price_prediction\data\Housing.csv'):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()

    # Drop duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values (drop or fill)
    df.dropna(inplace=True)

    # Separate categorical and numerical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.drop('price')

    # One-Hot Encode categorical columns
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(categorical_cols))

    # Scale numerical features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled, columns=numerical_cols)

    # Combine all features
    X = pd.concat([scaled_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    y = df['price']

    return X, y, encoder, scaler

# For testing the module directly
if __name__ == "__main__":
    df = load_data()
    X, y, encoder, scaler = preprocess_data(df)
    print("Processed data shape:", X.shape)
