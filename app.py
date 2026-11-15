import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

st.set_page_config(page_title="MODIS & VIIRS Harmonisation", layout="wide")
st.title("🔥 Harmonisation of MODIS & VIIRS Hotspots")

@st.cache_data
def load_and_harmonise():
    @st.cache_data
def load_data():
    cols = ['latitude','longitude','brightness','confidence','acq_date','acq_time','satellite','daynight']
    # Sirf kaam ke column load karo, aur 20,000 row hi kaafi hai demo ke liye
    modis = pd.read_csv("MODIS_C6_1_Global_24h.csv", usecols=cols, nrows=20000)
    viirs = pd.read_csv("J1_VIIRS_C2_Global_24h.csv", usecols=cols, nrows=20000)
    return modis, viirs

modis, viirs = load_data()
    viirs = viirs.rename(columns={'bright_ti4':'brightness', 'bright_ti5':'brightness_t5'})
    modis = modis.drop(columns=['version','bright_t31','satellite'], errors='ignore')
    viirs = viirs.drop(columns=['version','satellite'], errors='ignore')  
    modis['confidence_level'] = pd.cut(modis['confidence'], bins=[-1,30,80,100], labels=[0,1,2]).astype(int)
    viirs['confidence_level'] = viirs['confidence'].map({'low':0,'nominal':1,'high':2})
    modis['daynight'] = modis['daynight'].map({'D':0,'N':1})
    viirs['daynight'] = viirs['daynight'].map({'D':0,'N':1})
    modis['datetime'] = pd.to_datetime(modis['acq_date'].astype(str) + ' ' + modis['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
    viirs['datetime'] = pd.to_datetime(viirs['acq_date'].astype(str) + ' ' + viirs['acq_time'].astype(str).str.zfill(4), format='%Y-%m-%d %H%M')
    modis['source'] = 'MODIS'
    viirs['source'] = 'VIIRS'
    viirs['frp_calibrated'] = viirs['frp'] * 1.15
    modis['frp_calibrated'] = modis['frp']
    cols = ['latitude','longitude','brightness','frp_calibrated','confidence_level','daynight','datetime','source']
    combined = pd.concat([modis[cols], viirs[cols]], ignore_index=True)
    combined_sorted = combined.sort_values(['confidence_level','frp_calibrated'], ascending=False)
    combined_sorted['grid_lat'] = (combined_sorted['latitude']*100).round().astype(int)
    combined_sorted['grid_lon'] = (combined_sorted['longitude']*100).round().astype(int)
    harmonised = combined_sorted.drop_duplicates(['grid_lat','grid_lon'], keep='first')
    harmonised = harmonised.drop(columns=['grid_lat','grid_lon'])
    return combined, harmonised

combined, df = load_and_harmonise()

@st.cache_data
def train_model(_df):
    df2 = _df[_df['frp_calibrated'] < 200].copy()
    df2['hour'] = pd.to_datetime(df2['datetime']).dt.hour
    X = df2[['latitude','longitude','brightness','confidence_level','daynight','hour']]
    y = df2['frp_calibrated']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    df2['predicted_frp'] = model.predict(X)
    return df2, r2, model

df_model, r2, model = train_model(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Detections", f"{len(combined):,}")
col2.metric("After Harmonisation", f"{len(df):,}")
col3.metric("Duplicates Removed", f"{len(combined)-len(df):,}")
col4.metric("Model R2 Score", f"{r2:.3f}")

st.sidebar.header("Filter Map")
source = st.sidebar.multiselect("Select Satellite", options=df['source'].unique(), default=list(df['source'].unique()))
filtered = df[df['source'].isin(source)]
filtered = filtered.copy()
filtered['color'] = filtered['source'].apply(lambda x: [255,60,0,180] if x=='MODIS' else [0,120,255,180])
map_df = filtered.sample(min(5000, len(filtered)))
layer = pdk.Layer("ScatterplotLayer", data=map_df, get_position=["longitude","latitude"], get_radius=8000, get_fill_color="color", pickable=True)
view = pdk.ViewState(latitude=10, longitude=20, zoom=1.5)
st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "Source: {source}\nFRP: {frp_calibrated}"}))
st.subheader("🤖 Actual FRP vs Predicted FRP")
st.scatter_chart(data=df_model.sample(2000), x='frp_calibrated', y='predicted_frp', color='source')
st.markdown("**How We Harmonised:** Confidence 0-1-2 me badla, FRP x1.15 calibrate kiya, 1km Grid me duplicate hataye.")
