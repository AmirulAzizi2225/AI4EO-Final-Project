# Land Use and Land Cover (LULC) Classification on Sentinel-3 Satellite Imageries

## 📌 Problem Description

Accurate land cover classification is essential for monitoring environmental change, supporting urban planning, and informing policy decisions. Traditional classification methods can be labor-intensive and require extensive ground truth data. The goal of this project is to classify land cover over the southern United Kingdom region using Sentinel-3 satellite imagery and machine learning techniques. Both **unsupervised (K-Means)** and **supervised (Random Forest)** classification methods will be implemented, and their performance will be evaluated through visual and statistical comparisons.

K-Means clustering is an unsupervised machine learning algorithm used to group data points such as image pixels into a predefined number of clusters based on their feature similarity. In the context of remote sensing, each pixel is treated as a data point, with its spectral values (e.g., RGB, NDVI) serving as input features. The algorithm works by randomly initializing cluster centroids, assigning each pixel to the nearest centroid, recalculating centroids based on cluster membership, and repeating this process until convergence. The result is a classified image where each pixel belongs to one of the K clusters. However, these clusters are assigned arbitrary labels, so their real-world meaning (e.g., vegetation, water, urban) must be interpreted manually. While K-Means is simple, fast, and effective when classes are well-separated, it requires prior knowledge of the number of clusters and is sensitive to initial conditions and noise.

Random Forest is a supervised machine learning algorithm that is widely used for classification and regression tasks, including land cover classification in remote sensing. It works by constructing an ensemble of decision trees during training. Each tree is trained on a random subset of the training data (a process called bagging), and at each split in a tree, a random subset of features is considered. This randomness increases model diversity and reduces overfitting. When classifying a new data point such as a pixel in a satellite image, the prediction is made by aggregating the votes from all trees in the forest (typically via majority vote). In the context of land cover classification, Random Forest uses labeled training pixels with known class types (e.g., vegetation, water, urban) and learns to generalize from their spectral signatures. It is robust to noise, handles high-dimensional data well, and provides feature importance metrics. However, it requires quality training labels and may struggle if classes are spectrally similar or poorly represented in the training data.

See diagram below on the principle behind how the K-means clustering group the data: 
![K-means example](56854k-means-clustering.webp)

See diagram below on the principle behind how Random Forest classification works:
![Random-forest-example](Random-forest-concept.jpg)

---
## Prerequisites

The following steps need to be executed to run the code:

* Mounting Google Drive on Google Colab
  ```sh
  from google.colab import drive
  drive.mount('/content/drive')
  ```
* Install plug-in pandas
  ```sh
  !pip install requests shapely pandas
  ```
* Import important libraries
  ```sh
  import requests
  import pandas as pd
  import os
  ```

## 🛰️ 1. Data Collection

Level-1 OLCI imagery were acquired from **Sentinel-3** for the area of interest. The selected time frames include:
- **Image 1:** June–July 2023
- **Image 2:** Another scene with a similar time frame

```sh
def get_access_and_refresh_token(username, password):
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }
    response = requests.post(url, data=data)
    response.raise_for_status()
    tokens = response.json()
    return tokens["access_token"], tokens["refresh_token"]

# ESA Sentinel-3 credentials 
username = "# replace with your credentials"
password = "# replace with your credentials"
access_token, refresh_token = get_access_and_refresh_token(username, password)
```
The data are retrieved using the Copernicus Data Space Ecosystem API, with filtering by geographic bounding box and acquisition date. 

---

## 🧪 2. Preprocessing and Feature Extraction

### Normalized Difference Vegetation Index (NDVI)

**NDVI** = (NIR - RED) / (NIR + RED)

**Normalised Difference Vegetation Index (NDVI)** is a widely used remote sensing index that quantifies vegetation health and density using satellite imagery.

NDVI compares how much near-infrared (NIR) light vegetation reflects versus how much red light it absorbs.

*   Healthy plants absorb most red light for photosynthesis
*   Healthy plants reflect most NIR light due to cell structure
*   Water, soil, and urban areas reflect both differently

```sh
# Calculate NDVI
ndvi = (nir - red) / (nir + red)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(ndvi, cmap='RdYlGn')
plt.colorbar(label='NDVI')
plt.title("NDVI from Sentinel-3")
plt.axis("off")
plt.show()
```
NDVI visualisation of the region of interest:
![NDVI-image1](NDVI-image-1.jpg)

### RGB Composite

To help visually interpret the land surface, a true-colour RGB composite was created by stacking the Sentinel-3 red, green, and blue bands in the standard order.  The raw radiance values are normalized to the range [0, 1] to ensure consistent contrast and brightness for display. This visualization simulates how the scene would appear to the human eye, allowing us to clearly distinguish between vegetation (green areas), urban zones (gray or brown), and water (dark or bluish areas).

```sh
# Stack bands into RGB order: [Red, Green, Blue]
rgb = np.stack([
    red.values,
    green.values,
    blue.values
], axis=-1)

# Normalize to [0, 1] for display
rgb_min = np.nanmin(rgb)
rgb_max = np.nanmax(rgb)
rgb_norm = (rgb - rgb_min) / (rgb_max - rgb_min)

# Show RGB image
plt.figure(figsize=(10, 10))
plt.imshow(rgb_norm)
plt.title("RGB Composite from Sentinel-3")
plt.axis("off")
plt.show()
```
RGB Stacked Bands:
![rgb-image1](rgb-image-1.jpg)

### Preparing data for classification

Data must be made NumPy array and cleaned to remove NaN values
```sh
# Make sure NDVI is a NumPy array (ndvi) and cleaned (remove NaNs):
ndvi_clean = np.nan_to_num(ndvi.values.astype('float32'))
```
and reshape for clustering
```sh
# Reshape NDVI for clustering
h, w = ndvi_clean.shape
X = ndvi_clean.reshape(-1, 1)  # 1 feature (NDVI), many pixels
```

---

## 🤖 3. Classification Methods

### 🔹 Unsupervised Classification (K-Means)

K-Means clustering applied to the preprocessed imagery. It groups pixels based on spectral similarity without using ground truth. The result (`label_image`) assigns each pixel a cluster ID.

```sh
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=42)
labels = kmeans.fit_predict(X)
label_image = labels.reshape(h, w)
```

Since the output clusters are arbitrary, we manually **remapped the cluster IDs** to meaningful land cover classes (e.g., vegetation, water, cloud, urban) and assigned corresponding colors:
- Green: Vegetation
- Blue: Water
- Grey: Cloud
- Brown: Urban/Bare Soil

### 🔸 Supervised Classification (Random Forest)

We trained a **Random Forest** model using manually labeled pixels as training data. Once trained, the model was applied to the entire image to classify all pixels based on learned patterns. The result is stored as `classified_rf`.

This same workflow was applied to a second Sentinel-3 image to assess the **generalizability** of the model.

---

## 🖼️ 4. Visualisation

To facilitate comparison, both classification results were plotted side by side using consistent colormaps.

