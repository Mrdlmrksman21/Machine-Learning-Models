# === IMPORTS ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error

from google.colab import files
uploaded = files.upload()

# === LOAD DATA ===
df = pd.read_csv('smart_city_citizen_activity.csv')
df.columns = df.columns.str.strip()  # clean column names

# === Specify Target Columns ===
target_columns = [
    'Mode_of_Transport',   # classification
    'Carbon_Footprint_kgCO2', # regression
    'Gender'              # classification
]

# === Drop any unwanted feature columns ===
if 'Citizen_ID' in df.columns:
    df = df.drop('Citizen_ID', axis=1)  # remove IDs

# === Preprocess Categorical Columns ===
label_encoders = {}
for col in df.columns:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

# === Model Selection and Execution Loop ===
for target in target_columns:
    print(f"\nTarget Column: **{target}**")
    print("Choose technique:")
    print("1. Classification\n2. Regression\n3. Clustering\n4. Skip this column")
    choice = input("Enter option (1/2/3/4): ").strip()

    if choice == '4':
        print(f" Skipping {target}")
        continue

    # Prepare features/target
    if choice in ['1', '2']:
        X = df.drop(target_columns, axis=1)  # drop all targets from features
        y = df[target]

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Classification
        if choice == '1':
            print("Training Classification Model...")
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f" Accuracy: {acc:.4f}")
            print(classification_report(y_test, y_pred))

        # Regression
        elif choice == '2':
            print(" Training Regression Model...")
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            print(f" Mean Squared Error: {mse:.4f}")
            print(f" Example Predictions:\nActual: {np.round(y_test.values[:5], 2)}\nPredicted: {np.round(y_pred[:5], 2)}")

    # Clustering
    elif choice == '3':
        print(" Applying KMeans Clustering...")
        X = df.drop(target_columns, axis=1)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        print(f" Cluster Labels: {np.unique(clusters)}")
        print("Sample Cluster Assignments:")
        print(pd.DataFrame({'Cluster': clusters}).value_counts().head())

    else:
        print(" Invalid input. Skipping this target.")
