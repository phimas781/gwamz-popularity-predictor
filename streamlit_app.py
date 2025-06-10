import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load the trained model
model = joblib.load('spotify_popularity_model.pkl')

st.title("🎵 Gwamz Song Popularity Predictor")
st.write("Predict the Spotify popularity of a new Gwamz song based on its release date and album popularity.")

# User inputs
release_date = st.date_input("Release Date")
album_avg_popularity = st.number_input("Album Average Popularity", min_value=0, max_value=100, value=50)

if st.button("Predict Popularity"):
    # Calculate song age in days
    song_age_days = (pd.to_datetime('today') - pd.to_datetime(release_date)).days

    # Prepare input DataFrame
    new_song = pd.DataFrame([{
        'song_age_days': song_age_days,
        'album_avg_popularity': album_avg_popularity
    }])

    # Predict popularity
    predicted_popularity = model.predict(new_song)[0]
    st.success(f"🎤 Predicted Spotify Popularity: {predicted_popularity:.2f}")
