import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\user\Downloads\archive (1)\car data.csv", encoding='ISO-8859-1')

# Clean column names
df.columns = df.columns.str.strip().str.lower()
print("Cleaned Columns:", df.columns.tolist())

# Columns in your file: ['car_name', 'year', 'selling_price', 'present_price', 'driven_kms', 'fuel_type', 'selling_type', 'transmission', 'owner']

# Label encode categorical columns
le = LabelEncoder()
for col in ['car_name', 'fuel_type', 'selling_type', 'transmission', 'owner']:
    df[col] = le.fit_transform(df[col])

# Features & label
X = df.drop('selling_price', axis=1)
y = df['selling_price']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("✅ R² Score:", r2_score(y_test, y_pred))
print("✅ MAE:", mean_absolute_error(y_test, y_pred))

# Plot actual vs predicted
plt.figure(figsize=(8,5))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(True)
plt.tight_layout()
plt.show()

