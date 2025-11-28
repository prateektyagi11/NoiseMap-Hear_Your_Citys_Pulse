import os
import streamlit as st
import pandas as pd
import pydeck as pdk
import requests
from datetime import datetime

API_URL = os.getenv("API_URL", "http://api:8000")

st.set_page_config(layout="wide", page_title="NoiseMap — Hear Your City's Pulse")
st.title("NoiseMap — Hear Your City's Pulse")

@st.cache_data(ttl=60)
def get_recent(limit=2000):
    try:
        r = requests.get(f"{API_URL}/readings/recent?limit={limit}", timeout=10)
        r.raise_for_status()
        return pd.DataFrame(r.json())
    except Exception as e:
        st.error("Could not fetch data from API: " + str(e))
        return pd.DataFrame()

df = get_recent()

if df.empty:
    st.info("No data available. Use the /ingest endpoint to add readings.")
else:
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    min_db = int(df['db_level'].min())
    max_db = int(df['db_level'].max())
    db_filter = st.sidebar.slider("Filter by dB level", min_db, max_db, (min_db, max_db))
    df = df[(df['db_level'] >= db_filter[0]) & (df['db_level'] <= db_filter[1])]

    st.subheader("Heatmap")
    midpoint = (float(df['lat'].median()), float(df['lon'].median()))
    layer = pdk.Layer(
        "HeatmapLayer",
        data=df,
        get_position='[lon, lat]',
        get_weight='db_level',
        radiusPixels=60,
    )
    view_state = pdk.ViewState(latitude=midpoint[0], longitude=midpoint[1], zoom=11)
    r = pdk.Deck(layers=[layer], initial_view_state=view_state)
    st.pydeck_chart(r)

    st.subheader("Noise over time (1H avg)")
    ts = df.groupby(pd.Grouper(key='timestamp', freq='1H')).db_level.mean().reset_index()
    ts = ts.set_index('timestamp')
    st.line_chart(ts)
