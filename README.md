# Land Use and Land Cover (LULC) Classification on Sentinel-3 Satellite Imageries

## 📌 Problem Statement

Accurate land cover classification is essential for monitoring environmental change, supporting urban planning, and informing policy decisions. Traditional classification methods can be labor-intensive and require extensive ground truth data. In this project, we aim to classify land cover over the Greater London area using Sentinel-3 satellite imagery and machine learning techniques. We implement both **unsupervised (K-Means)** and **supervised (Random Forest)** classification methods, and evaluate their performance through visual and statistical comparisons.

---

## 🛰️ 1. Data Collection

We acquired Level-1 OLCI imagery from **Sentinel-3** for the Greater London area. The selected time frames include:
- **Image 1:** June–July 2023
- **Image 2:** Another scene from a different date for consistency testing

The data was retrieved using the Copernicus Data Space Ecosystem API, with filtering by geographic bounding box and acquisition date.

---

## 🧪 2. Preprocessing and Feature Extraction

We computed the **Normalized Difference Vegetation Index (NDVI)** to enhance the visibility of vegetation:

\[
\text{NDVI} = \frac{NIR - RED}{NIR + RED}
\]

This index, along with RGB bands, was used as input features for classification.

---

## 🤖 3. Classification Methods

### 🔹 Unsupervised Classification (K-Means)

K-Means clustering was applied to the preprocessed imagery. It groups pixels based on spectral similarity without using ground truth. The result (`label_image`) assigns each pixel a cluster ID.

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

