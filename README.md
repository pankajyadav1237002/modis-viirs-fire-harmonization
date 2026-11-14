# 🔥 MODIS & VIIRS Fire Hotspot Harmonization System

An open-source geospatial tool and interactive web application built to harmonize, analyze, and visualize active fire hotspot data from the MODIS (Moderate Resolution Imaging Spectroradiometer) and VIIRS (Visible Infrared Imaging Radiometer Suite) satellite sensors. 

Developed as part of the NASA Space Apps Challenge, this project addresses the critical challenge of cross-satellite data fusion to deliver unified, reliable wildfire and agricultural fire intelligence.

---

## 🚀 Key Features

1. Data Harmonization: Aligns spatial resolutions, temporal frequencies, and confidence metrics between MODIS and VIIRS datasets to eliminate discrepancies.
2. Machine Learning Integration: Utilizes robust regression models (e.g., `RandomForestRegressor`) to predict and normalize fire radiative power (FRP) and confidence levels.
3. Interactive Streamlit Dashboard: Provides a user-friendly web interface for real-time filtering, spatial mapping, and comparative analytics.
4. Geospatial Visualization: Renders interactive maps and statistical charts using Python geospatial libraries (`GeoPandas`, `Folium`, `Plotly`).

---

## 🛠️ Tech Stack

1. Language: Python 3.11+
2. Data Processing & ML: Pandas, NumPy, Scikit-Learn
3. Geospatial Analysis: GeoPandas, Shapely
4. Web Interface: Streamlit
5. Visualization: Matplotlib, Seaborn, Plotly

---

## 📂 Project Structure

```text
├── app.py                # Main Streamlit web application
├── requirements.txt      # Project dependencies
├── data/                 # Raw and cleaned MODIS/VIIRS datasets
├── models/               # Trained machine learning models (.pkl)
└── README.md             # Project documentation
