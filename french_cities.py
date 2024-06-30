# %%
# Import necessary libraries
# The libraries are listed by their appearence in the code
import pandas as pd
import geopandas as gpd
from unidecode import unidecode
import folium
from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
import matplotlib.pyplot as plt
import numpy as np

# from sklearn.cluster import KMeans
import statsmodels.api as sm
import seaborn as sns
from fcapy.lattice import ConceptLattice
from fcapy.context import FormalContext
from fcapy.visualizer import LineVizNx

# %%
"""
About the data :
The dataset covers a wide range of socio-economic, environmental (Nature),
and health-related indicators of 100 cities in France from the year 2005.
the variables/features are partitioned like this : Economics : Chomage, ChomageJeunes,
ChomageLong, EvoluEmploiCree, Activite, EmploiFeminin, EmploiCommune,
DefaillEntreprise, SalaireAnnuel, ImpotRevenu, ImpotFortune, Imposables,
MetreCarreAncien, TaxeHabitation, FoncierBati, MetreCubeEau, EvolDemographique,
Vieillissement, AttiranceGlobale, AttiranceActifs, Proprietaires, LogtSup4pieces,
LogtInsalubre, LogtVacant, LogtConstruction

Risk Factors : Criminalite, EvolutionCrimes, SecuriteRoutiere, Inondations,
TerrainsPollues, UsinesRisques, MortaliteInfantile, MortaliteCancerPoumon,
MortaliteAlcool, DecesInfarctus, TauxSuicide, MortaliteGlobale, TailleClassesPrimaires,
Retard6eme, Retard3eme, RetardTerminale

Nature : Mer, Ski, Soleil, Pluie, Temperature, MarcheAPied

Culture : Musees, Cinema, MonumHistoriques, PretLivres, RestaurDistingues,
Presse, Etudiants
"""
# %%
"""
Meta data :
A tsv file (tab separated values) that contains 55 variables (columns)
and 100 observations(rows).

1. Ville: The city or municipality.
2. Chomage: Overall unemployment rate.
3. ChomageJeunes: Youth unemployment rate (typically for ages 15-24).
4. ChomageLong: Long-term unemployment rate (usually for those unemployed for over a year).
5. EvoluEmploiCree: Employment growth or jobs created.
6. Activite: Labor force participation rate.
7. EmploiFeminin: Female employment rate.
8. EmploiCommune: Employment rate within the commune.
9. DefaillEntreprise: Business failure rate or number of business bankruptcies.
10. SalaireAnnuel: Average annual salary.
11. Criminalite: Crime rate.
12. EvolutionCrimes: Change in crime rate over a specified period.
13. SecuriteRoutiere: Road safety statistics (e.g., accidents, fatalities).
14. Inondations: Frequency or risk of flooding.
15. TerrainsPollues: Number or area of polluted lands.
16. UsinesRisques: Number of high-risk factories (e.g., chemical plants).
17. MortaliteInfantile: Infant mortality rate.
18. MortaliteCancerPoumon: Mortality rate from lung cancer.
19. MortaliteAlcool: Mortality rate related to alcohol consumption.
20. DecesInfarctus: Mortality rate from heart attacks (infarctions).
21. TauxSuicide: Suicide rate.
22. MortaliteGlobale: Overall mortality rate.
23. TailleClassesPrimaires: Average class size in primary schools.
24. Retard6eme: Proportion of students held back in 6th grade.
25. Retard3eme: Proportion of students held back in 9th grade.
26. RetardTerminale: Proportion of students held back in the final year of high school.
27. MetreCarreAncien: Average price per square meter of older properties.
28. TaxeHabitation: Residential tax rate.
29. FoncierBati: Property tax rate on built property.
30. MetreCubeEau: Cost per cubic meter of water.
31. Proprietaires: Homeownership rate.
32. LogtSup4pieces: Proportion of homes with more than four rooms.
33. LogtInsalubre: Proportion of substandard housing.
34. LogtVacant: Proportion of vacant housing units.
35. LogtConstruction: Number or rate of new housing constructions.
36. Mer: Qualitative data that describes numbers of beaches.
37. Ski: Has a ski resort or No.
38. Soleil: Average annual sunshine (hours).
39. Pluie: Average annual rainfall (mm).
40. Temperature: Average annual temperature.
41. MarcheAPied: Walkability score or pedestrian-friendliness.
42. Musees: Number of museums.
43. Cinema: Number of cinemas.
44. MonumHistoriques: Number of historical monuments.
45. PretLivres: Number of books loaned (e.g., by libraries).
46. RestaurDistingues: Number of distinguished restaurants (e.g., Michelin-starred).
47. Presse: index of number of press outlets or newspapers.
48. Etudiants: Index of number of students.
49. ImpotRevenu: Income tax rate.
50. ImpotFortune: Wealth tax rate.
51. Imposables: Proportion of taxable households.
52. EvolDemographique: Demographic growth rate.
53. Vieillissement: Aging index (ratio of elderly to young population).
54. AttiranceGlobale: Overall attractiveness score (a composite index).
55. AttiranceActifs: Attractiveness for working-age population.
"""
# %%
# Read the tsv file and set ville (cities) as index :
df = pd.read_csv("../Datasets/French_cities_2005.csv", sep="\t", index_col="Ville")
df.index = df.index.str.lower()
# df.Ville = df.Ville.str.lower()
# %%
"""
Let's learn a lil bit about the french cities here
"""
# %%
# The french cities in this data set are :
print(f"The french cities available in this dataset are : \n {list(df.index)}")
# %%
"""
Read the data we scrapped using Geopandas
"""
# %%
cities = gpd.read_file("../Code/regions.geojson")
# %%
"""
format the new data set to make it match the data set we have so we can merge the
two datasets :
- We'll unidecode it : remove the accents in the french language like é, è and ô...
- Remove dashes between
- Lower case the values of the column in common
"""


# %%
def remove_accents_and_dashes(word):
    # Remove accents using unidecode
    word_normalized = unidecode(word)
    # Remove dashes
    word_normalized = word_normalized.replace("-", "")
    word_normalized = word_normalized.lower()
    return word_normalized


cities["nom"] = [remove_accents_and_dashes(word) for word in cities["nom"]]
cities
# %%
"""
Now we will merge the two Datasets:
"""
# %%
all_cities = pd.merge(cities, df, left_on="nom", right_index=True, how="inner")
all_cities = all_cities.reset_index()
all_cities.shape
# %%
"""
- Okay so we got 88% information from the Data we scrapped, which is not bad
- A better practice would be making sure we got the longitude and latitude
of all of 100 cities
- We'll get the centroid of each city so we can add a marker on the map.
"""
# %%
mean_latitudes = []
mean_longitudes = []

# Loop through each geometry entry in the 'geometry' column
for geom in all_cities["geometry"]:
    if geom.is_valid:
        # Calculate the centroid of the geometry
        centroid = geom.centroid
        mean_lon, mean_lat = centroid.x, centroid.y
    else:
        mean_lon, mean_lat = None, None

    mean_longitudes.append(mean_lon)
    mean_latitudes.append(mean_lat)

# Add these lists as new columns in your GeoDataFrame
all_cities["mean_latitude"] = mean_latitudes
all_cities["mean_longitude"] = mean_longitudes
# %%
"""
Now we will create a map and highlight the country of France and mark the cities
"""
# %%
map = folium.Map([46.716671, 3], zoom_start=5)

cities_map = folium.GeoJson(
    cities,
    style_function=lambda feature: {
        "fillColor": "blue",
        "color": "white",
        "weight": 0.5,
    },
    name="Map of cities in France",
    zoom_on_click=True,
).add_to(map)

for i in range(len(all_cities)):
    folium.Marker(
        location=[
            all_cities["mean_latitude"][i],
            all_cities["mean_longitude"][i],
        ],  # Convert to (latitude, longitude)
        popup=all_cities["nom"][i].capitalize(),
        tooltip=all_cities["nom"][i].capitalize(),
    ).add_to(map)
# map
# %%
"""
- Now we will be interested in making a HAC (hierarchical agglomerative clustering)
of this Dataset to find cities that are similar based on the indicators we have...
- We will start that by doing a quick feature engineering where we will standardize
the data : by doing that Standardization allows for the rebalancing of scales and the
elimination of the effect of units of each variable. This enables us to compare
variables and individuals effectively, regardless of their original scales,
It also helps us avoid mutual compensation when calculating Euclidean distances
to measure similarity/dissimilarity for hierarchical clustering
"""

# %%
# We will set the citites as index of our dataframe
df_std = pd.DataFrame(preprocessing.scale(df), columns=df.columns, index=df.index)
# %%
"""
- The principle of hierarchical clustering is to easily group or aggregate according
to a predefined similarity criterion, which is expressed in the form of a distance
matrix. This matrix shows the distance between each pair of individuals.
Two identical observations/individuals will have a distance of zero.

If we consider two points \( p \) and \( q' \) in \( \mathbf{R}^{n} \) with respective
coordinates \( (x_1,...,x_n) \) and \( (y_1,...,y_n) \), their Euclidean distance
is given by the formula:
$$
\[
\left\lVert p - q' \right\rVert = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}
\]
$$
- We perform hierarchical clustering of French cities using Ward's index
as a measure of aggregating clusters. we will be using different Cluster linkage
and compare them
- We will do that using scipy library and create a simple function to make the code
much cleaner
"""


# %%
def linked_method(dataframe, method):
    return linkage(
        dataframe,
        method=method,
        optimal_ordering=True,
    )


# %%
def plot_dendrogram(data, method, threshold=1000):
    plt.figure(figsize=(14, 8))
    dendrogram(
        data,
        orientation="top",
        labels=df_std.index.values,
        get_leaves=True,
        distance_sort="ascending",
        color_threshold=threshold,
    )
    plt.axhline(
        y=threshold, color="black", linestyle="--", label="Horizontal Line at y=5"
    )
    plt.title(f"Dendrogramm of the linkeage criterion: {method} ")
    plt.xlabel("Cities")
    plt.ylabel("Euclidean distance")
    return plt.show()


# %%
# Ward linkage (Minimum Increase of Sum of Squares)
ward_linkeage = linked_method(df_std, "ward")
plot_dendrogram(ward_linkeage, "Ward")
# %%
# Maximum or complete-linkage clustering
max_linkage = linked_method(df_std, "complete")
plot_dendrogram(max_linkage, "Max")
# %%
min_linkage = linked_method(df_std, "single")
plot_dendrogram(min_linkage, "Min")
# %%
# Average Linkeage
average_linkeage = linked_method(df_std, "average")
plot_dendrogram(average_linkeage, "Average")
# %%
# Centroid Linkage
centroid_linkage = linked_method(df_std, "centroid")
plot_dendrogram(centroid_linkage, "Centroid")

# %%
"""
- From this dendrogram obtained from the hierarchy indicated by the Ward index,
we notice the potential to have fairly homogeneous classes. We observe different
drops at the level of the aggregation cost, and the classes seem compact.
- The maximum linkeage criterion gave similar results to Ward's
- We notice that the minimum linkeage criterion does not have sufficiently homogeneous
classes, but the classes seem continuous due to the relatively small gap between
them. This does not serve to partition the individuals in a homogeneous manner
but rather in a continuous way. This index can be used to detect continuity or
contiguity in the data, but that is not the case in this work.
- So we will continue our work with the Ward criterion...
"""
# %%
"""
- To determine the addequate number of classes to make clusters using Ward index,
we can use a barplot of to observe how the values of ward's series and how
it is decreasing, a good indicator where we should cut our dendrogram is when we could
notice a peak in this bar plot
"""


# %%
def diff_series(linkage_matrix):
    data = pd.DataFrame(
        linkage_matrix,
        columns=[
            "n° partition",
            "n° partiton_2",
            "Aggregation_cost",
            "n°obs orginal dans partition",
        ],
    )
    data = data.sort_values(by=data.columns[2], ascending=False)
    diff = abs(data["Aggregation_cost"].diff())
    data = data.join(diff, rsuffix="_Diff")
    return data


ward_diff_series = diff_series(ward_linkeage)
ward_diff_series


# %%
def plot_diff_series(diff_series):
    plt.figure(figsize=(11, 7))
    plt.bar(
        np.arange(0, diff_series["Aggregation_cost_Diff"].size),
        diff_series["Aggregation_cost_Diff"],
        width=0.6,
        color="red",
    )
    # Ajouter des titres et des étiquettes
    plt.title("Bar plot")
    plt.xlabel("Series of differences")
    plt.ylabel("Values ")
    # Afficher le diagramme
    return plt.show()


plot_diff_series(ward_diff_series)

# %%
"""
The largest drops encourage us to choos k=3 or k=7. However, for simplicity
in interpretation, we will choose k = 3, with a cut-off value of 25;
the aggregation cost we have chosen to partition our individuals into 3 classes like
it is showed below :
"""
# %%
plot_dendrogram(ward_linkeage, "ward", threshold=25)
# %%
"""
- So we get three classes/Clusters :
Sure, here's the text without the LaTeX format:

- First class:
Boulogne-Billancourt, Neuilly-sur-Seine, Paris, Rueil-Malmaison, Saint-Germain-en-Laye, Versailles.
  - 8 cities form a class in the region of "western Paris".

- Second class:
Ajaccio, Antibes, Arles, Avignon, Bastia, Béziers, Cannes, Marseille, Nice, Nîmes,
Perpignan, Sète, Toulon.
  - With 13 cities, the second class represents "the cities of southern France".

- Third class: Agen, Aix-en-Provence, Albi, ... Saint-Nazaire,Saint-Quentin,
Sarcelles, Strasbourg, Tarbes, Toulouse, Tours, Troyes, Valence, Valenciennes, Vannes, Vichy, Villeurbanne.
  - With 81 cities, this class describes "the rest of the cities in France".
"""
# %%
"""
- Now we will go through a deeper analysis inside these three classes:
- Treating the variable "cities class" as Y
- We'll calculate the R² of each quantitative variable with the variable "cities
class" which will help us to find us the variables that are correlated with the
city's class.
"""
# %%
# We'll create three clusters
cut = cut_tree(ward_linkeage, n_clusters=3)


# One-hot coding the the variables Y : cities classes
def one_hot_encoding(clusters):
    data = pd.DataFrame(clusters, dtype="category", index=df_std.index)
    data = data.rename(columns={0: "class"})
    data = pd.get_dummies(data, columns=["class"], dtype=int)
    return data


df_3 = one_hot_encoding(cut)
df_3
# %%
"""
We'll calculate the weight/mean of each class
"""
# %%
weights = df_3.mean()
weights


# %%
# Now we will get three data sets for each attribute of the our variable cities classes
def filter_df_by_class(df, attr):
    data = pd.DataFrame(df[df[attr] > 0], columns=[attr])
    return df_std.loc[data.index]


# %%
df_class0 = filter_df_by_class(df_3, "class_0")
df_class1 = filter_df_by_class(df_3, "class_1")
df_class2 = filter_df_by_class(df_3, "class_2")


# %%
# Calculate the R² of each variable
# We will simply use an Linear model to find it :
def Calculate_R2(Quali_var, Quanti_var, sorted=False, show=5):
    data = pd.DataFrame(0, index=Quanti_var.columns, columns=["R_squared"])
    for column in Quanti_var.columns:
        model = sm.OLS(Quanti_var[column], sm.add_constant(Quali_var))

        results = model.fit()
        data.loc[column] = results.rsquared * 100
    return data.sort_values(by="R_squared", ascending=sorted).head(show)


R2_m1 = Calculate_R2(df_3, df_std)
R2_m1
# %%
# The 10 first features that are explains the disparity/diffrences between the cities in %
round(R2_m1, ndigits=2).sort_values(by="R_squared", ascending=False).head(10) * 100
# %%
# The variables least correlated with the city class variable are
round(R2_m1 * 100, 3).sort_values(by="R_squared", ascending=True).head(8)
# %%
# We can also calculate the R2 of each variable manually without passing by the linear model :
R2_m2 = pd.DataFrame(
    100
    * (
        (df_class0.mean() ** 2) * weights[0]
        + (df_class1.mean() ** 2) * weights[1]
        + (df_class2.mean() ** 2) * weights[2]
    ),
    columns=["R²"],
)
R2_m2
# We basically find the same results and we could save a lot of computation !!!
# %%
# We can get the total R² of the variable which is the mean value of all  R² of each variable :
R2_m2.mean()
# %%
"""
Thus the variable cities classes explains around 20% of the variability between the different
french cities.
- Next we will recreate the Dataset and add the class of each city
"""
# %%
classes_cities = {0: "Rest_of_France", 1: "South_of_France", 2: "west_Of_Paris"}


def create_df_with_classes(df, clusters, classes):
    data = df.copy()
    data["Class"] = clusters.astype("int")
    data["Class"] = data["Class"].map(classes)
    return data


df_c = create_df_with_classes(df_std, cut, classes_cities)
df_c
# %%
"""
The box plot of a variable \( x^j \) conditioned on the class variable:
A box plot, also known as a box-and-whisker plot, is a graphical representation of
the distribution of a dataset. It provides a summary of key statistical measures such
as the minimum, first quartile (Q1), median (Q2), third quartile (Q3), and maximum.
It is often used to compare differences between groups in analysis of variance.
Outliers can also be detected using a box plot.
Let's take the example of the variable 'SalaireAnnuel':
"""


# %%
def box_plot_class_var(dataframe, var):
    sns.boxplot(dataframe, x="Class", y=var)
    return plt.show()


# %%
box_plot_class_var(df_c, "SalaireAnnuel")
# %%
"""
- The box plot of the West Paris class is higher than the other two classes,
indicating a difference in annual salaries between the classes. It also explains
the dispersion of the data points by the values in the West Paris class,
where the variance of the West Paris class is higher than that of the other two
classes: thus, heterogeneity of variances can be observed among the classes.
- It is also notable that Neuilly-sur-Seine is an outlier in the West Paris class.
By using box plots, we can compare the different city classes and interpret their
results for each variable \( x^j \)."
"""
# %%
# This noticeable variance is due to the city of west of Paris (known for having
# rich families and celebrities as residents)
df_class2[["SalaireAnnuel"]]

# %%
box_plot_class_var(df_c, "Cinema")
# For the variable Cinema we can see there is not a significant variance between classes
# %%
"""
Now we will try to optimize the HAC by using k-means simply by :
- Calculating the center of gravity of each class found by HAC
- Initialize the k-means Algorithm with the center of gravities
- Find new optimized classes
"""


# %%
# The center of gravity of each class can be found :
def find_center_of_gravity(df_ohe, df):
    Center_of_gravity = np.linalg.inv(df_ohe.T @ df_ohe) @ (df_ohe.T @ df)
    return Center_of_gravity


Center_of_gravity_C3 = find_center_of_gravity(df_3, df_std)
Center_of_gravity_C3
# %%
# Usually i'd use this line of code to fit the k-means algorithm
# KMV2 = KMeans(n_clusters=3,init=CentreC3).fit(df_std)
# But the algorithm used in python didn't do a lot of changes so i used the on in R
# Here is the code to fit K-means in R:
# mIC2Y = as.matrix(IC2Y)
#
# mY = as.matrix(Y)
# CentresC2 = solve(t(mIC2Y)%*%mIC2Y)%*%t(mIC2Y)%*%mY
#
#  K-means à partir de ces centres initiaux:
# KMV2 = kmeans(Y, CentresC2)
#
#  La variable de classe ainsi produite est dans:
# apr_km = KMV2$cluster
#
# Obtenir les noms de la répartition des villes pour chaque groupes avant kmeans
# k1 = rownames(X)[apr_km == 1]
# k2 = rownames(X)[apr_km == 2]
# k3 = rownames(X)[apr_km == 3]
# %%
# I copied the result directly from R :
df_km = [
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    2,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    2,
    0,
    1,
    1,
    0,
    0,
    2,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    2,
    0,
    0,
    0,
    2,
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    2,
    0,
    0,
]
df_km = np.array(df_km)

# %%
# Create a new data set to encode the variable Cities classes
df_km = one_hot_encoding(df_km)
df_km
# %%
df_class0_opt = filter_df_by_class(df_km, "class_0")
df_class1_opt = filter_df_by_class(df_km, "class_1")
df_class2_opt = filter_df_by_class(df_km, "class_2")

# %%
R2_Opt = Calculate_R2(df_km, df_std, show=df_std.shape[0])
R2_Opt.mean()
# %%
"""
After optimization with K-means, we added 4 individuals to the South of France class:
Aix-en-Provence, Carcassonne, Montpellier, and Castres.
We also calculated the \( R^2 \) of the new partition, which is equal to
\( R^2 = 0.196674 \), an increase of \( 0.41\% \).

These are the results of classifying the 100 French cities into 3 classes,
which are not sufficiently homogeneous in number. This heterogeneity can be explained
by analyzing the data by themes to gain more information about each class,
which contributes to the dispersion of these three classes.
- Our next analysis will focus on exploring the similarities within classes
for each theme, which will be the subject of our work.
"""
# %%
"""
- Thematic classification involves balancing themes by finding a way to move from overall
similarity to partial similarity for greater logic and naturally classifying cities in
a manner based on reality. A city can thus belong to multiple classes. For this,
we will perform a classification by theme.
"""
# %%
"""
- We will start by the first theme : Economy
"""
# %%
df_eco = df_std[
    [
        "Chomage",
        "ChomageJeunes",
        "ChomageLong",
        "EvoluEmploiCree",
        "Activite",
        "EmploiFeminin",
        "EmploiCommune",
        "DefaillEntreprise",
        "SalaireAnnuel",
        "ImpotRevenu",
        "ImpotFortune",
        "Imposables",
        "MetreCarreAncien",
        "TaxeHabitation",
        "FoncierBati",
        "MetreCubeEau",
        "EvolDemographique",
        "Vieillissement",
        "AttiranceGlobale",
        "AttiranceActifs",
        "Proprietaires",
        "LogtSup4pieces",
        "LogtInsalubre",
        "LogtVacant",
        "LogtConstruction",
    ]
]
# %%
ward_linkeage_eco = linked_method(df_eco, "ward")
plot_dendrogram(ward_linkeage_eco, "Ward for theme economy")
# %%
diff_series_econ = diff_series(ward_linkeage_eco)
plot_diff_series(diff_series_econ)
# %%
plot_dendrogram(ward_linkeage_eco, "Ward for theme economy", threshold=18)
# %%
cut_eco_clusters = cut_tree(ward_linkeage_eco, n_clusters=3)
# One hot encoding of our data set
df_ohe_eco = one_hot_encoding(cut_eco_clusters)
df_ohe_eco
# %%
df_class0_eco = filter_df_by_class(df_ohe_eco, "class_0")
df_class1_eco = filter_df_by_class(df_ohe_eco, "class_1")
df_class2_eco = filter_df_by_class(df_ohe_eco, "class_2")
# %%
R2_eco = Calculate_R2(df_ohe_eco, df_eco)
R2_eco
# %%
R2_eco_total = Calculate_R2(df_ohe_eco, df_eco, show=df_eco.shape[0]).mean()
R2_eco_total
# %%
classes_cities_eco = {0: "Low_econ_class", 1: "Middle_econ_class", 2: "Rich_econ_class"}
df_c_eco = create_df_with_classes(df_eco, cut_eco_clusters, classes_cities_eco)
df_c_eco
# %%
box_plot_class_var(df_c_eco, "SalaireAnnuel")
box_plot_class_var(df_c_eco, "MetreCarreAncien")
# %%
Center_of_gravity_eco = find_center_of_gravity(df_ohe_eco, df_eco)
Center_of_gravity_eco
# %%
km_eco = [
    1,
    1,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    2,
    1,
    2,
    2,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    3,
    1,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    2,
    1,
    2,
    2,
    1,
    1,
    1,
    2,
    1,
    1,
    2,
    2,
    1,
    1,
    1,
    1,
    3,
    1,
    2,
    2,
    2,
    1,
    3,
    1,
    1,
    2,
    1,
    2,
    1,
    1,
    1,
    3,
    1,
    1,
    2,
    3,
    2,
    2,
    2,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    1,
    2,
    1,
    1,
    3,
    1,
    1,
]
km_eco = np.array(km_eco)
# %%
df_km_eco = one_hot_encoding(km_eco)

df_km_eco
# %%
df_class0_eco_opt = filter_df_by_class(df_km_eco, "class_1")
df_class1_eco_opt = filter_df_by_class(df_km_eco, "class_2")
df_class2_eco_opt = filter_df_by_class(df_km_eco, "class_3")
# %%
R2_eco_opt = Calculate_R2(df_km_eco, df_eco)
# %%
R2_eco_opt_total = Calculate_R2(df_km_eco, df_eco, show=df_eco.shape[0]).mean()
R2_eco_opt_total
# %%
"""
Here's the translated text without LaTeX format:

- The first class ("Agen", "AixEnProvence", "Amiens", "Angers", "Angouleme", "Annecy",
"Auxerre", "Avignon", ...., "Strasbourg", "Tarbes", "Toulouse", "Tours", "Troyes",
"Valence", "Valenciennes", "Vannes", "Villeurbanne") is composed of 65 individuals
before K-means.

- The second class ("Ajaccio", "Albi", "Antibes", "Arles", "Bastia", "Beziers", ...,
"Marseille", "Montauban", "Montlucon", "Nice", "Nimes", "Perpignan", "Quimper",
"SaintEtienne", "SaintMalo", "SaintNazaire", "SaintQuentin", "Sete", "Toulon", "Vichy")
is composed of 29 individuals before K-means.

- The third class ("BoulogneBillancourt", "NeuillySurSeine", "Paris", "RueilMalmaison",
"SaintGermainEnLaye", "Versailles") is composed of 6 individuals.

Now let's move on to the optimization of these classes using K-means: we calculated the
centroids of each class in the economic theme, then we ran the K-means algorithm and
found the following results:

- The first class went from 65 individuals to 56 after optimization,
losing some individuals. This class can be described as disadvantaged or
low class.

- The second class went from 29 individuals to 38 individuals, thus undergoing
optimization as it was enriched by 9 new individuals. This class can be described as
moderate or middle class.

- The third class did not undergo any optimization and remains stable. This class
can be described as wealthy or rich class.
"""
# %%
"""
- Now the second theme : risk with 16 features
"""
# %%
# 25 features
df_risk = df_std[
    [
        "Criminalite",
        "EvolutionCrimes",
        "SecuriteRoutiere",
        "Inondations",
        "TerrainsPollues",
        "UsinesRisques",
        "MortaliteInfantile",
        "MortaliteCancerPoumon",
        "MortaliteAlcool",
        "DecesInfarctus",
        "TauxSuicide",
        "MortaliteGlobale",
        "TailleClassesPrimaires",
        "Retard6eme",
        "Retard3eme",
        "RetardTerminale",
    ]
]
# %%
ward_linkeage_risk = linked_method(df_risk, "ward")
plot_dendrogram(ward_linkeage_risk, "Ward for theme risk")
# %%
diff_series_risk = diff_series(ward_linkeage_risk)
plot_diff_series(diff_series_risk)
# %%
plot_dendrogram(ward_linkeage_risk, "Ward for theme risk", threshold=16.5)
# %%
cut_risk_clusters = cut_tree(ward_linkeage_risk, n_clusters=2)
# %%
df_ohe_risk = one_hot_encoding(cut_risk_clusters)
df_ohe_risk
# %%
df_class0_risk = filter_df_by_class(df_ohe_risk, "class_0")
df_class1_risk = filter_df_by_class(df_ohe_risk, "class_1")

# %%
R2_risk = Calculate_R2(df_ohe_risk, df_risk)
R2_risk
# %%
R2_risk_total = Calculate_R2(df_ohe_risk, df_risk, show=df_risk.shape[0]).mean()
R2_risk_total
# %%
classes_cities_risk = {0: "Safe", 1: "Risky"}
df_c_risk = create_df_with_classes(df_risk, cut_risk_clusters, classes_cities_risk)
df_c_risk
# %%
box_plot_class_var(df_c_risk, "MortaliteGlobale")
box_plot_class_var(df_c_risk, "MortaliteAlcool")
# %%
Center_of_gravity_risk = find_center_of_gravity(df_ohe_risk, df_risk)
Center_of_gravity_risk
# %%
# Optimisation with k-means
km_risk = [
    1,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    1,
    2,
    2,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    1,
    2,
    2,
    2,
    1,
    2,
    1,
    2,
    2,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    2,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    1,
    1,
    2,
    1,
    2,
    1,
    1,
    2,
    1,
]
km_risk = np.array(km_risk)
# %%
df_km_risk = one_hot_encoding(km_risk)
# %%
df_class0_risk_opt = filter_df_by_class(df_km_risk, "class_1")
df_class1_risk_opt = filter_df_by_class(df_km_risk, "class_2")
# %%
R2_risk_opt = Calculate_R2(df_km_risk, df_risk)
R2_risk_opt
# %%
R2_risk_opt_total = Calculate_R2(df_km_risk, df_risk, show=df_risk.shape[0]).mean()
R2_risk_opt_total
# %%
"""
- The first class ("Agen", "AixEnProvence", "Ajaccio", "Albi", "Angers", "Angouleme",
..., "Sarcelles", "Tarbes", "Toulon", "Tours", "Valence", "Vannes", "Versailles", "Vichy",
"Villeurbanne") is composed of three-fourths of the individuals before K-means.

- The second class ("Amiens", "Beauvais", "Belfort", "Bordeaux", "Calais",
"CharlevilleMezieres", "Colmar", "Dunkerque", "Epinal", ..., "Mulhouse", "Nancy",
"Reims", "Rouen", "SaintQuentin", "Sete", "Strasbourg", "Toulouse", "Troyes", "Valenciennes") is composed of one-fourth of the individuals before K-means.

Now let's continue by optimizing these classes with the K-means algorithm, where we
initialized the K-means algorithm with the centroids of the classes obtained from
the hierarchical clustering.

- The first class went from 75 individuals to 65 after optimization, losing some
individuals. This class can be described as safe.

- The second class went from 25 individuals to 35 individuals, thus undergoing
as it was enriched by 10 new individuals. This class can be described as risky.
"""
# %%
"""
Now we'll treat the theme nature : with 6 features
"""
# %%
df_nature = df_std[["Mer", "Ski", "Soleil", "Pluie", "Temperature", "MarcheAPied"]]
# %%
ward_linkeage_nature = linked_method(df_nature, "ward")
plot_dendrogram(ward_linkeage_nature, "Ward for theme risk")
# %%
diff_series_nature = diff_series(ward_linkeage_nature)
plot_diff_series(diff_series_nature)
# %%
plot_dendrogram(ward_linkeage_nature, "Ward for them nature", threshold=16)
# %%
cut_nature_clusters = cut_tree(ward_linkeage_nature, n_clusters=2)
# %%
df_ohe_nature = one_hot_encoding(cut_nature_clusters)
df_ohe_nature
# %%
df_class0_nature = filter_df_by_class(df_ohe_nature, "class_0")
df_class1_nature = filter_df_by_class(df_ohe_nature, "class_1")
# %%
R2_nature = Calculate_R2(df_ohe_nature, df_nature)
R2_nature
# %%
R2_nature_total = Calculate_R2(df_ohe_nature, df_nature, show=df_nature.shape[0]).mean()
R2_nature_total
# %%
classes_cities_nature = {0: "Inland_city", 1: "Coastal_city"}
df_c_nature = create_df_with_classes(
    df_nature, cut_nature_clusters, classes_cities_nature
)
df_c_risk
# %%
box_plot_class_var(df_c_nature, "Pluie")
box_plot_class_var(df_c_nature, "Soleil")
# %%
Center_of_gravity_nature = find_center_of_gravity(df_ohe_nature, df_nature)
Center_of_gravity_nature
# %%
# Optimization of HCA with k-means
km_nature = [
    1,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    1,
    2,
    2,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    2,
    2,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    1,
    2,
    1,
    1,
    1,
    1,
    1,
]
km_nature = np.array(km_nature)
# %%
df_km_nature = one_hot_encoding(km_nature)

# %%
df_class0_nature_opt = filter_df_by_class(df_km_nature, "class_1")
df_class1_nature_opt = filter_df_by_class(df_km_nature, "class_2")
# %%
R2_nature_opt = Calculate_R2(df_km_nature, df_nature)
R2_nature_opt
# %%
R2_nature_opt_total = Calculate_R2(
    df_km_nature, df_nature, show=df_nature.shape[0]
).mean()
R2_nature_opt_total

# %%
"""
Here's the translated text without LaTeX format:

We optimize the partition using K-means with the centroids of the two classes obtained
with hierarchical clustering and obtain the following two classes:

- The first class ("Agen", "Albi", "Amiens", "Angers", "Angouleme", ..., "Strasbourg",
"Tarbes", "Toulouse", "Tours", "Troyes", "Valenciennes", "Vannes", "Versailles", "Vichy",
"Villeurbanne") is composed of 83 individuals before and after K-means optimization;
this class of cities can be described as: Inland Cities

- The second class ("AixEnProvence", "Ajaccio", "Antibes", "Arles", "Avignon",
"Bastia", "Beziers", "Cannes", "Carcassonne", "Marseille", "Montpellier", "Nice",
"Nimes", "Perpignan", "Sete", "Toulon", "Valence") is composed of 17 individuals
before and after K-means optimization; this class can be described according to the
nature theme as: Coastal Cities
"""
# %%
"""
Now we will hop on the theme Culture with 7 features
"""
# %%
df_culture = df_std[
    [
        "Musees",
        "Cinema",
        "MonumHistoriques",
        "PretLivres",
        "RestaurDistingues",
        "Presse",
        "Etudiants",
    ]
]
# %%
ward_linkeage_culture = linked_method(df_culture, "ward")
plot_dendrogram(ward_linkeage_culture, "Ward for theme Culture")
# %%
diff_series_culture = diff_series(ward_linkeage_culture)
plot_diff_series(diff_series_culture)
# %%
plot_dendrogram(ward_linkeage_culture, "Ward for them culture", threshold=12)
# %%
cut_culture_clusters = cut_tree(ward_linkeage_culture, n_clusters=3)
# %%
df_ohe_culture = one_hot_encoding(cut_culture_clusters)
df_ohe_culture
# %%
df_class0_culture = filter_df_by_class(df_ohe_culture, "class_0")
df_class1_culture = filter_df_by_class(df_ohe_culture, "class_1")
df_class2_culture = filter_df_by_class(df_ohe_culture, "class_2")
# %%
R2_culture = Calculate_R2(df_ohe_culture, df_culture)
R2_culture
# %%
R2_culture_total = Calculate_R2(
    df_ohe_culture, df_culture, show=df_culture.shape[0]
).mean()
R2_culture_total
# %%
classes_cities_culture = {0: "mediocre_culture", 1: "Rich_culture", 2: "Paris"}
df_c_culture = create_df_with_classes(
    df_culture, cut_culture_clusters, classes_cities_culture
)
df_c_culture
# %%
box_plot_class_var(df_c_culture, "RestaurDistingues")
box_plot_class_var(df_c_culture, "Musees")
box_plot_class_var(df_c_culture, "MonumHistoriques")
# %%
Center_of_gravity_culture = find_center_of_gravity(df_ohe_culture, df_culture)
Center_of_gravity_culture
# %%
# Optimization of HCA with k-means
km_culture = [
    2,
    1,
    2,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    1,
    1,
    1,
    1,
    2,
    2,
    1,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    2,
    2,
    1,
    1,
    2,
    1,
    1,
    2,
    1,
    1,
    2,
    2,
    2,
    1,
    2,
    1,
    2,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
    1,
    1,
    2,
    2,
    2,
    2,
    1,
    2,
    2,
]
km_culture = np.array(km_culture)
# %%
df_km_culture = one_hot_encoding(km_culture)
# %%
df_class0_culture_opt = filter_df_by_class(df_km_culture, "class_1")
df_class1_culture_opt = filter_df_by_class(df_km_culture, "class_2")
# %%
R2_culture_opt = Calculate_R2(df_km_culture, df_culture)
R2_culture_opt
# %%
R2_culture_opt_total = Calculate_R2(
    df_km_culture, df_culture, show=df_culture.shape[0]
).mean()
R2_culture_opt_total
# %%
"""
- The first class ("Agen", "Ajaccio", "Albi", "Angouleme", "Annecy", "Antibes", "Arles",
"Bastia", "Bayonne", "Beauvais", "Belfort", ..., "SaintEtienne","Troyes", "Valence",
"Valenciennes", "Vannes", "Vichy") is composed of 67 individuals before K-means.
This class can be described as cities with modest culture.

- The second class ("AixEnProvence", "Amiens", "Angers", "Auxerre", "Avignon", "Besancon",
"Bordeaux", "Caen", "Cannes", "ClermontFerrand", ..., "Montpellier", "Nancy", "Nantes",
"Orleans", "Paris", "Perpignan", "Poitiers", "Reims", "Rennes", "SaintDenis",
"SaintGermainEnLaye", "Strasbourg", "Toulouse", "Tours") is composed of 33 individuals
before K-means. This class can be described as cultural cities.

- Third class consist of Paris being a stand out because it is one of the most culturally
rich cities in the whole world.

- Despite applying the K-means algorithm, there was no big difference in optimization besides
getting paris to join the cultural cities.
"""
# %%
"""
The thematic classification continues with a particular focus on the themes "economy"
and "risk," consisting of 25 and 16 variables respectively. A division into three
classes—disadvantaged, moderate, and wealthy—was carried out for the "economy" theme,
while a division into two classes—predictable and precarious—was applied to the "risk"
theme, with significant optimization. However, for the themes "nature" and "culture,"
although the classes were identified, no real optimization was observed after applying
the K-means algorithm. The detailed results, dendrograms, and tables provide an in-depth
analysis of the clusters of individuals before and after K-means for each theme.
"""

# %%
"""
We will treat all themes as latent qualitative thematic variables with their modalities
to enrich our work, moving towards a more logical and formal classification using formal
concept analysis.
"""
# %%
"""
Now, we focus on the part of partial resemblance within the thematic classification framework.
This involves the complete one hot encoding table of the four thematic variables, where
each variable has its modalities.
"""
# %%
df_km_eco = df_km_eco.rename(
    columns={
        "class_1": "Low_econ_class",
        "class_2": "Middle_econ_class",
        "class_3": "Rich_econ_class",
    }
)
df_km_nature = df_km_nature.rename(
    columns={"class_1": "Inland_city", "class_2": "Coastal_city"}
)
df_km_culture = df_km_culture.rename(
    columns={"class_1": "mediocre_culture", "class_2": "Rich_culture"}
)
df_km_risk = df_km_risk.rename(columns={"class_1": "Safe", "class_2": "Risky"})

# %%

themes_ohe = (df_km_eco.join(df_km_culture).join(df_km_nature).join(df_km_risk)).astype(
    bool
)
themes_ohe
# %%
# Construct a FormalContext from a binarized pandas dataframe
fca_data = FormalContext.from_pandas(themes_ohe)
# %%
# Construct a Concept Lattice :
concept_lattice = ConceptLattice.from_context(fca_data)

# %%
print(f"We have {len(concept_lattice)} concept in this lattice")
"""
- A lattice represent a ccouple of intent and extent where each intent is a feature
and extent is an observation (city in our case here)
"""
# %%
# We'll visualize all the nodes and try to get some good information
fig, ax = plt.subplots(figsize=(40, 20))
vsl = LineVizNx(
    node_label_font_size=20,
    node_size=1000,
    flg_axes=True,
    node_alpha=50,
    flg_drop_bottom_concept=True,
)
vsl.draw_concept_lattice(
    concept_lattice,
    ax=ax,
    flg_node_indices=True,
    max_nex_intent_count=3,
    max_new_extent_count=1,
    # Prefix shows count of features of cities
    flg_new_intent_count_prefix=True,
    # Prefix shows count of cities that have a feature
    flg_new_extent_count_prefix=True,
)
ax.set_xlim(ax.get_xlim()[0], ax.get_xlim()[1])
plt.title("whatever", size=45)
plt.tight_layout()
plt.show()


# %%
def get_intent_extent_of_node(i):
    return print(
        f"{(concept_lattice[i].extent)}: \n characterized by \n {(concept_lattice[i].intent)} "
    )


# %%
get_intent_extent_of_node(64)
# %%
get_intent_extent_of_node(65)
# %%
get_intent_extent_of_node(60)
# %%
get_intent_extent_of_node(66)
# %%
"""
We can also get the child and ancestors of each node thus we can get more
information about partial resemblance between cities
"""
# %%
concept_lattice.descendants(15)
# %%
concept_lattice.ancestors(15)
# %%
concept_lattice.children(15)
# %%
concept_lattice.parents(15)

# %%
"""
In conclusion, this study provided an in-depth analysis of the data through the classification
of individuals and variables. The classification of individuals revealed significant clusters,
highlighting global and partial similarities and differences among them throughout different
dimensions and spaces.

"""