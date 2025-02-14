[car data.csv](https://github.com/user-attachments/files/18802052/car.data.csv)
# Task-3
Car price Prediction with Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Drop 'Car_Name' as it's not useful for prediction
df = df.drop(columns=["Car_Name"])

# Define features and target variable
X = df.drop(columns=["Selling_Price"])
y = df["Selling_Price"]

# Preprocessing for numerical and categorical features
num_features = ["Year", "Present_Price", "Driven_kms", "Owner"]
cat_features = ["Fuel_Type", "Selling_type", "Transmission"]

num_transformer = StandardScaler()
cat_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_transformer, num_features),
        ("cat", cat_transformer, cat_features)[car data.csv](https://github.com/user-attachments/files/18796462/car.data.csv)

    ]
)

# Define model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

mae, rmse
