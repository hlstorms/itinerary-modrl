import pandas as pd
import json

# Load the JSON file
with open("historical_weather_data.json", "r") as file:
    weather_data = json.load(file)

# Convert the JSON data into a DataFrame
df = pd.DataFrame(weather_data)

# 1. Map the 'rain' column to "rain" and "no_rain"
df["rain"] = df["rain"].apply(lambda x: "rain" if x else "no_rain")

# 2. Create a 'temp' column with temperatures categorized into 5 uniform batches
# Use pd.cut to create 5 bins with numerical labels
df["temp"] = pd.cut(
    df["avg_temp_c"].astype(float),  # Convert temperatures to float
    bins=5,  # Divide into 5 uniform bins
    labels=[0, 1, 2, 3, 4]  # Numeric labels for the bins
)

# 3. Keep only the necessary columns: 'date', 'temp', and 'rain'
df = df[["date", "temp", "rain"]]

# Display the resulting DataFrame
print(df.head())

# Save the DataFrame to a CSV file (optional)
df.to_csv("data/weather_data.csv", index=False)
print("Transformed data saved to 'transformed_weather_data.csv'.")
