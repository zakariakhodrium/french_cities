# %%
import requests
import bs4
import json

# %%
"""
- This data is scrapped from this website: https://france-geojson.gregoiredavid.fr/
- The website doesn't show the names of departments so I had to scrape it from the GitHub repository:
  https://github.com/gregoiredavid/france-geojson/tree/master/regions
- I excluded the Overseas Departments of France alongside the value None to get a clean set of items:
  [None, "guadeloupe", "martinique", "guyane", "la-reunion", "mayotte"]
- Then I will fetch the data from the website in the first bullet point and add it to the second key of a dictionary
  required for the geojson file, meanwhile, the first pair of key-value is fixed.
- Finally, I will save the file in geojson format.
"""

# %%
# URL of the GitHub repository containing the list of regions
url_repo = "https://github.com/gregoiredavid/france-geojson/tree/master/regions"

# Send a GET request to the repository URL
get_regions = requests.get(url_repo)

# Parse the HTML content of the page using BeautifulSoup
soup = bs4.BeautifulSoup(get_regions.text, "html.parser")

# Find the section containing the repository content
repo_content = soup.find(class_="repository-content")

# %%
# Initialize an empty list to hold the names of the regions
regions = []

# List of regions to exclude (Overseas Departments of France and None)
excluded_regions = [None, "guadeloupe", "martinique", "guyane", "la-reunion", "mayotte"]

# Iterate over all 'a' tags in the repository content
for link in repo_content.find_all("a"):
    # Get the 'title' attribute of each 'a' tag
    title = link.get("title")
    # If the title is not in the excluded regions list, add it to the regions list
    if title not in excluded_regions:
        regions.append(title)

# Remove duplicates by converting the list to a set and back to a list
regions = list(set(regions))

# %%
# Initialize the dictionary for the geojson file with a fixed 'type' and an empty 'features' list
dictionnary = {"type": "FeatureCollection", "features": []}

# Iterate over each region to fetch its geojson data
for i in regions:
    # Construct the URL for the geojson file of the current region
    url = f"https://france-geojson.gregoiredavid.fr/repo/regions/{i}/arrondissements-{i}.geojson"

    # Send a GET request to the geojson file URL
    response = requests.get(url)

    # Parse the JSON response
    data = response.json()

    # Extend the 'features' list in the dictionary with the features from the current geojson file
    dictionnary["features"].extend(data["features"])

# Write the final dictionary to a file in geojson format
with open("regions.geojson", "w") as outfile:
    json.dump(dictionnary, outfile, indent=1)
