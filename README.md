# French cities
```
french_cities
├── Data
│   ├── French_cities_2005.csv
│   └── regions.geojson
├── README.md
├── french_cities.ipynb
├── french_cities.py
└── get_data.py
```
**PS:** I wrote the main code french_cities.py using ipython then converted it to "french_cities.ipynb" using [p2j](https://pypi.org/project/p2j/) for those who are more familiar with Jupyter notebooks

In this repository, you will find what I did :
- Wrote a code to fetch and scrap data of French cities': latitude and longitude from [this website](https://france-geojson.gregoiredavid.fr/) and this [repository](https://github.com/gregoiredavid/france-geojson/tree/master/regions) and wrote it in a ".geojson" file (**PS: you don't need to run this file**)
- Visualized the cities on an interactive map
- Analyzed 100 French cities based on 50 different features throughout four themes: economy, risk, nature and culture.
- Applied HCA (hierarchical agglomerative clustering) to cluster the cities based on 50 variables
- Optimized the result of HCA clusters with the help of k-means using the centroid of clusters obtained in HCA as initial points for k-means algorithm
- Applied HCA on different themes and also optimized the clusters with k-means...
- Did a [formal concept analysis](https://en.wikipedia.org/wiki/Formal_concept_analysis) on the whole data sets by creating new latent descriptive variables based on the cluster's results to understand and analyze the partial resemblance between the cities and get sub-clusters that share different features

**Libraries used :**
- Beautiful soup and requests to fetch and scrap the data
- Pandas, Geopandas to manipulate data
- [Unidecode](https://pypi.org/project/Unidecode/) for formatting
- Sklearn for preprocessing
- Folium to visualize the data on a map
- Scipy for HCA and dendrogram visualization
- [fcapy](https://pypi.org/project/fcapy/) for formal concept analysis

**NB: I will probably refactor the code in 'fenchc_cities.ipynb' to make it less redundant**