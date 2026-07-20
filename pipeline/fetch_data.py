import os
import json
import requests

# Define root and data directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "raw_cards.json")

# Ensure the data directory actually exists
os.makedirs(DATA_DIR, exist_ok=True)

print("Fetching card data from YGOPRODeck API... This might take a moment.")

try:
    response = requests.get("https://db.ygoprodeck.com/api/v7/cardinfo.php")
    # This line checks if the download was successful (HTTP Status 200)
    response.raise_for_status() 
    
    # Parse the JSON response body
    payload = response.json()
    # The API wraps its list of cards inside a top-level key called 'data'
    raw_cards = payload.get("data", [])
    
    print(f"Successfully downloaded {len(raw_cards)} total cards.")

except requests.RequestException as e:
    print(f"Error fetching data from API: {e}")
    exit(1)

# Filter out cards that do not belong to an archetype
filtered_cards = []
for card in raw_cards:
    if "archetype" in card:
        filtered_cards.append(card)

print(f"Filtered down to {len(filtered_cards)} cards with valid archetypes.")

# Save our curated list to a local JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(filtered_cards, f, indent=4)

print(f"Saved filtered data to {OUTPUT_FILE}")