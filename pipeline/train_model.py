import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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

# Initialize and train the Random Forest
# Reduced n_estimators to 50 to decrease memory usage on Vercel
model = RandomForestClassifier(n_estimators=50)
model.fit(X_train, y_train)

# Ensure the output directory exists
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

# Save the model and encoders so our web app can use them
joblib.dump(model, os.path.join(BASE_DIR, "models", "archetype_model.joblib"), compress=3)
joblib.dump(label_encoders, os.path.join(BASE_DIR, "models", "label_encoders.joblib"), compress=3)
joblib.dump(target_encoder, os.path.join(BASE_DIR, "models", "target_encoder.joblib"), compress=3)

print("Model trained and saved to models/!")