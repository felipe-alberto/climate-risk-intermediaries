# Under Development by Felipe Verastegui at Columbia IEOR
# For more information see sites.google.com/view/felipealberto. 


import pandas as pd
import requests

# Load your payouts
payouts = pd.read_csv("data/raw/payouts.csv")

# Get WB country list
url = "https://api.worldbank.org/v2/country/all/indicator/NE.CON.GOVT.ZS?format=json&per_page=20000"
data = requests.get(url).json()[1]

wb = pd.DataFrame(data)
wb["country"] = wb["country"].apply(lambda x: x["value"] if isinstance(x, dict) else x)

wb_countries = wb["country"].dropna().unique()

# Find unmatched countries
unmatched = sorted(set(payouts["Country"].dropna()) - set(wb_countries))

print("Unmatched countries:")
print(unmatched)

print(wb[wb["country"].str.contains("Bahamas", case=False, na=False)][["country", "countryiso3code"]].drop_duplicates())
print(wb[wb["country"].str.contains("Anguilla", case=False, na=False)][["country", "countryiso3code"]].drop_duplicates())