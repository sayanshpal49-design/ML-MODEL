import urllib.request
import csv
import json
import numpy as np
import os

# 1. Download dataset
url = "https://raw.githubusercontent.com/aiplanethub/Datasets/master/Bengaluru_House_Data.csv"
csv_path = "Bengaluru_House_Data.csv"
print("Downloading dataset...")
try:
    urllib.request.urlretrieve(url, csv_path)
    print("Download complete.")
except Exception as e:
    print(f"Download failed: {e}")
    raise e

# 2. Parse CSV
rows = []
location_counts = {}

with open(csv_path, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    # Header mapping:
    # 0: area_type, 1: availability, 2: location, 3: size, 4: society, 5: total_sqft, 6: bath, 7: balcony, 8: price
    for r in reader:
        if not r or len(r) < 9:
            continue
        loc = r[2].strip() if r[2] else "Other"
        size_str = r[3].strip() if r[3] else ""
        sqft_str = r[5].strip() if r[5] else ""
        bath_str = r[6].strip() if r[6] else ""
        balcony_str = r[7].strip() if r[7] else ""
        price_str = r[8].strip() if r[8] else ""
        
        # Parse BHK
        bhk = 2.0
        if size_str:
            parts = size_str.split()
            if parts and parts[0].isdigit():
                bhk = float(parts[0])
                
        # Parse sqft
        sqft = None
        if sqft_str:
            if '-' in sqft_str:
                try:
                    parts = sqft_str.split('-')
                    sqft = (float(parts[0].strip()) + float(parts[1].strip())) / 2.0
                except:
                    pass
            else:
                try:
                    sqft = float(sqft_str)
                except:
                    pass
        if sqft is None:
            continue
            
        # Parse bath
        bath = 2.0
        if bath_str:
            try:
                bath = float(bath_str)
            except:
                pass
                
        # Parse balcony
        balcony = 1.0
        if balcony_str:
            try:
                balcony = float(balcony_str)
            except:
                pass
                
        # Parse price (Target is in Lakhs Rupees)
        price = None
        if price_str:
            try:
                price = float(price_str)
            except:
                pass
        if price is None:
            continue
            
        rows.append({
            'location': loc,
            'bhk': bhk,
            'sqft': sqft,
            'bath': bath,
            'balcony': balcony,
            'price': price
        })
        location_counts[loc] = location_counts.get(loc, 0) + 1

# 3. Select top 20 locations
sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
top_locations = [loc for loc, count in sorted_locations[:20]]

# 4. Build feature matrix X and target y
X = []
y = []

for r in rows:
    loc = r['location']
    features = [
        1.0, # intercept
        r['bhk'],
        r['sqft'],
        r['bath'],
        r['balcony']
    ]
    # One-hot encode locations
    for top_loc in top_locations:
        features.append(1.0 if loc == top_loc else 0.0)
    
    X.append(features)
    y.append(r['price'])

X = np.array(X)
y = np.array(y)

# 5. Fit Linear Regression model using NumPy least squares
w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

# 6. Compute predictions and residuals
y_pred = np.dot(X, w)
residuals = y - y_pred
std_err = float(np.std(residuals))

# 7. Save model to JSON
model_data = {
    'weights': w.tolist(),
    'top_locations': top_locations,
    'std_err': std_err
}

os.makedirs('model_files', exist_ok=True)
with open('model_files/indian_model.json', 'w') as f:
    json.dump(model_data, f, indent=2)

print("SUCCESS: Trained linear regression model on Bengaluru House Prices.")
print(f"Number of samples: {len(y)}")
print(f"Residual standard error: {std_err:.2f} Lakhs")
