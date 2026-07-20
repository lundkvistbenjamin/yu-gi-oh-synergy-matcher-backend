import os
import json
import pandas as pd

# Define root and data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT_FILE = os.path.join(DATA_DIR, "raw_cards.json")
OUTPUT_FILE = os.path.join(DATA_DIR, "clean_cards.csv")

# Load the raw data
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    cards = json.load(f)

# Convert the raw JSON array into a Pandas DataFrame
df = pd.DataFrame(cards)

# Select only the columns that represent our "features" and our "target"
columns_to_keep = ["name", "type", "race", "atk", "def", "level", "attribute", "archetype"]
df = df[columns_to_keep]

print(f"Loaded DataFrame shape: {df.shape}")

# Fill missing numerical fields
df['atk'] = df['atk'].fillna(-1)
df['def'] = df['def'].fillna(-1)
df['level'] = df['level'].fillna(0)

# Fill missing categorical text fields
df['attribute'] = df['attribute'].fillna('NONE')

# Force data types to integers so decimals don't float around
df['atk'] = df['atk'].astype(int)
df['def'] = df['def'].astype(int)
df['level'] = df['level'].astype(int)

# Save to a CSV file without the default Pandas index numbers
df.to_csv(OUTPUT_FILE, index=False)
print(f"Cleaned dataset saved successfully to {OUTPUT_FILE}")