import pandas as pd
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Get the directory of the current script to build absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, "data", "clean_cards.csv")

# Load the data we just cleaned
df = pd.read_csv(CSV_PATH)

# We drop 'name' because it's unique to every card and won't help us 
# identify an archetype pattern.
features = df.drop(['name', 'archetype'], axis=1)
target = df['archetype']

# Create encoders for our text columns
label_encoders = {}
for col in ['type', 'race', 'attribute']:
    le = LabelEncoder()
    features[col] = le.fit_transform(features[col])
    label_encoders[col] = le  # Save this to decode later!

# Encode our target (the archetype names)
target_encoder = LabelEncoder()
target = target_encoder.fit_transform(target)

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2)

# Initialize and train the Logistic Regression model
model = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Ensure the output directory exists
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# EXTRACT RAW MATHEMATICAL DATA FOR ULTRA-LIGHTWEIGHT DEPLOYMENT
# We convert the Scikit-Learn matrices directly into basic Python lists
model_data = {
    "classes": target_encoder.classes_.tolist(),
    "intercept": model.intercept_.tolist(),
    "coefficients": model.coef_.tolist(),
    "encoders": {
        "type": label_encoders['type'].classes_.tolist(),
        "race": label_encoders['race'].classes_.tolist(),
        "attribute": label_encoders['attribute'].classes_.tolist()
    }
}

# Write directly to a plain text JSON file (a few kilobytes total)
with open(os.path.join(BASE_DIR, "models", "model_data.json"), "w") as f:
    json.dump(model_data, f)

print("Lightweight model coefficients successfully exported to models/model_data.json!")