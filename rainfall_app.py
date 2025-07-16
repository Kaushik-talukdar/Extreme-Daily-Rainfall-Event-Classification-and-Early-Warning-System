import streamlit as st
import joblib
import pandas as pd
import numpy as np
import datetime

# --- Your st.set_page_config() call should be here ---
st.set_page_config(page_title="Extreme Rainfall Predictor", layout="centered")


# --- 1. Load the Trained Model ---
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model():
    try:
        model_filename = 'final_extreme_rainfall_lgbm_model.joblib'
        loaded_model = joblib.load(model_filename)
        return loaded_model
    except FileNotFoundError:
        st.error(f"Error: Model file '{model_filename}' not found. "
                 "Please ensure the model file is in the same directory as this script.")
        st.stop() # Stop the app if model is not found
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop() # Stop the app if model loading fails

loaded_model = load_model()

# --- 2. Define Expected Features ---
# This list must EXACTLY match the features the model was trained on (111 features).
expected_features = [
    'District LGD Code', 'Latitude', 'Longitude', 'Manual Daily Rainfall (mm)',
    'Year', 'Month', 'Day', 'DayOfWeek', 'DayOfYear', 'Month_sin', 'Month_cos',
    'DayOfYear_sin', 'DayOfYear_cos', 'rainfall_lag_1d', 'rainfall_lag_2d',
    'rainfall_lag_3d', 'rainfall_lag_7d', 'rainfall_lag_14d', 'rainfall_lag_30d',
    'rainfall_rolling_mean_3d', 'rainfall_rolling_sum_3d', 'rainfall_rolling_mean_7d',
    'rainfall_rolling_sum_7d', 'rainfall_rolling_mean_14d', 'rainfall_rolling_sum_14d',
    'neighbor_avg_rainfall_lag_1d_50km',
    'Station_Baghmari Tea State', 'Station_Bahari High School', 'Station_Balbala GP Office Campus',
    'Station_Barnadi GD Site', 'Station_Barpeta  E & D Office', 'Station_Barpeta Road Inv. WR Divn ',
    'Station_Basbari E&D Colony', 'Station_Basugaon ENDQ', 'Station_Beki', 'Station_Bengtal State Dispensary',
    'Station_Bessamora', 'Station_Bhabanipur PHE', 'Station_Bijni', 'Station_Birjhara Tea Garden',
    'Station_Boko Circle Office', 'Station_Chapar PHC', 'Station_Chief Eng Office Campus',
    'Station_Deosri F Vill', 'Station_Dhansiri GD Site', 'Station_Dotma R/G Site',
    'Station_Golokganj RG Site', 'Station_Hatisar F. Vill', 'Station_Irrg. Divn Off. Bongaigaon',
    'Station_Kachugaon', 'Station_Khagrabari', 'Station_Kokilamukh', 'Station_Kokrajhar Circuit House',
    'Station_Kulsi Forest Office', 'Station_Manikpur DFO Compound', 'Station_Mathanguri',
    'Station_Nalbari Office Campus', 'Station_Panbari R/G Site', 'Station_Pandu R/G Site',
    'Station_Patacharkuchi PWD', 'Station_Rajabari', 'Station_Roumari State Dispensary',
    'Station_Rupahighat v/d Compound', 'Station_Sarukhetri High School', 'Station_Saudatari v/d Compound',
    'Station_Silghat', 'Station_Srijongram', 'Station_Ukiam',
    'District_BONGAIGAON', 'District_Baksa', 'District_Chirang', 'District_DARRANG', 'District_DHUBRI',
    'District_GOALPARA', 'District_GOLAGHAT', 'District_JORHAT', 'District_KAMRUP', 'District_KOKRAJHAR',
    'District_Kamrup Metropolitan', 'District_Majuli', 'District_NAGAON',
    'District_NALBARI', 'District_SONITPUR',
    'Tehsil_BARPETA', 'Tehsil_BOKAKHAT', 'Tehsil_BOKO', 'Tehsil_Baksa', 'Tehsil_Bhawanipur', 'Tehsil_Bongaigaon',
    'Tehsil_CHAMARIA', 'Tehsil_CHANDRAPUR', 'Tehsil_CHAPAR', 'Tehsil_CHHAYGAON', 'Tehsil_DOTOMA', 'Tehsil_Dangtol',
    'Tehsil_GOLOKGANJ', 'Tehsil_JORHAT EAST', 'Tehsil_JORHAT WEST', 'Tehsil_Jalah', 'Tehsil_KOKRAJHAR',
    'Tehsil_Krishnai', 'Tehsil_Majuli', 'Tehsil_Manikpur', 'Tehsil_NAGAON', 'Tehsil_NALBARI', 'Tehsil_Rani',
    'Tehsil_SARUPETA', 'Tehsil_SRIJANGRAM', 'Tehsil_Sidli Chirang', 'Tehsil_TEZPUR', 'Tehsil_UDALGURI'
]

# --- 3. Streamlit UI Layout ---


st.title("🌧️ Extreme Daily Rainfall Event Predictor")
st.markdown("""
This application predicts the likelihood of an **Extreme Daily Rainfall Event**
at a specific location in Assam, India.
""")

st.info("""
**Note on Data Input:** In a full-fledged system, many features like lagged rainfall,
rolling statistics, and nearby rainfall averages would be automatically calculated
from historical data. For this demo, these values are set to reasonable defaults.
You can adjust the 'Manual Daily Rainfall (mm)' and other numerical inputs
to see how the prediction changes.
""")

# --- Input Fields ---
st.header("Location and Date Information")

col1, col2 = st.columns(2)
with col1:
    district_lgd_code = st.number_input("District LGD Code", min_value=1, value=311, help="LGD Code for the district (e.g., 311 for Chirang)")
    latitude = st.number_input("Latitude", min_value=24.0, max_value=28.0, value=26.5000, format="%.4f")
    longitude = st.number_input("Longitude", min_value=89.0, max_value=97.0, value=90.6500, format="%.4f")
with col2:
    prediction_date = st.date_input("Date for Prediction", datetime.date(2023, 7, 15))
    manual_daily_rainfall = st.number_input("Manual Daily Rainfall (mm) for today", min_value=0.0, value=50.0, help="Rainfall recorded for the current day at this station.")

st.subheader("Select Station, District, and Tehsil")

# These lists should ideally come from your training data's unique values
# For now, using the ones inferred from your feature list
stations = [
    'Baghmari Tea State', 'Bahari High School', 'Balbala GP Office Campus',
    'Barnadi GD Site', 'Barpeta  E & D Office', 'Barpeta Road Inv. WR Divn ',
    'Basbari E&D Colony', 'Basugaon ENDQ', 'Beki', 'Bengtal State Dispensary',
    'Bessamora', 'Bhabanipur PHE', 'Bijni', 'Birjhara Tea Garden',
    'Boko Circle Office', 'Chapar PHC', 'Chief Eng Office Campus',
    'Deosri F Vill', 'Dhansiri GD Site', 'Dotma R/G Site',
    'Golokganj RG Site', 'Hatisar F. Vill', 'Irrg. Divn Off. Bongaigaon',
    'Kachugaon', 'Khagrabari', 'Kokilamukh', 'Kokrajhar Circuit House',
    'Kulsi Forest Office', 'Manikpur DFO Compound', 'Mathanguri',
    'Nalbari Office Campus', 'Panbari R/G Site', 'Pandu R/G Site',
    'Patacharkuchi PWD', 'Rajabari', 'Roumari State Dispensary',
    'Rupahighat v/d Compound', 'Sarukhetri High School', 'Saudatari v/d Compound',
    'Silghat', 'Srijongram', 'Ukiam'
]
districts = [
    'BARPETA', 'BONGAIGAON', 'Baksa', 'Chirang', 'DARRANG', 
    'DHUBRI', 'GOALPARA', 'GOLAGHAT', 'JORHAT', 'KAMRUP', 'KOKRAJHAR', 
    'Kamrup Metropolitan', 'Majuli', 'NAGAON', 'NALBARI', 'SONITPUR'
]
tehsils = [
    'BARPETA', 'BOKAKHAT', 'BOKO', 'Baksa', 'Bhawanipur', 'Bongaigaon', 'CHAMARIA',
    'CHANDRAPUR', 'CHAPAR', 'CHHAYGAON', 'DOTOMA', 'Dangtol', 'GOLOKGANJ',
    'JORHAT EAST', 'JORHAT WEST', 'Jalah', 'KOKRAJHAR', 'Krishnai', 'Majuli',
    'Manikpur', 'NAGAON', 'NALBARI', 'Rani', 'SARUPETA', 'SRIJANGRAM',
    'Sidli Chirang', 'TEZPUR', 'UDALGURI'
]

selected_station = st.selectbox("Station", options=stations, index=stations.index('Bijni'))
selected_district = st.selectbox("District", options=districts, index=districts.index('Chirang'))
selected_tehsil = st.selectbox("Tehsil", options=tehsils, index=tehsils.index('BOKO')) # Assuming Boko for Bijni, adjust if needed

# --- Prediction Button ---
if st.button("Predict Extreme Rainfall"):
    # --- Prepare Data for Prediction ---
    input_data = {}

    # Core numerical features
    input_data['District LGD Code'] = district_lgd_code
    input_data['Latitude'] = latitude
    input_data['Longitude'] = longitude
    input_data['Manual Daily Rainfall (mm)'] = manual_daily_rainfall

    # Date-related features
    input_data['Year'] = prediction_date.year
    input_data['Month'] = prediction_date.month
    input_data['Day'] = prediction_date.day
    input_data['DayOfWeek'] = prediction_date.weekday() # Monday=0, Sunday=6
    input_data['DayOfYear'] = prediction_date.timetuple().tm_yday
    input_data['Month_sin'] = np.sin(2 * np.pi * prediction_date.month / 12)
    input_data['Month_cos'] = np.cos(2 * np.pi * prediction_date.month / 12)
    input_data['DayOfYear_sin'] = np.sin(2 * np.pi * input_data['DayOfYear'] / 365)
    input_data['DayOfYear_cos'] = np.cos(2 * np.pi * input_data['DayOfYear'] / 365)

    # --- Mock/Default values for complex features ---
    # These values are illustrative. In a real system, they would be calculated
    # from historical data relative to the prediction_date and selected_station.
    input_data['rainfall_lag_1d'] = manual_daily_rainfall * 0.8 # Example: 80% of today's rainfall
    input_data['rainfall_lag_2d'] = manual_daily_rainfall * 0.6
    input_data['rainfall_lag_3d'] = manual_daily_rainfall * 0.4
    input_data['rainfall_lag_7d'] = manual_daily_rainfall * 0.3
    input_data['rainfall_lag_14d'] = manual_daily_rainfall * 0.2
    input_data['rainfall_lag_30d'] = manual_daily_rainfall * 0.1

    input_data['rainfall_rolling_mean_3d'] = (input_data['Manual Daily Rainfall (mm)'] + input_data['rainfall_lag_1d'] + input_data['rainfall_lag_2d']) / 3
    input_data['rainfall_rolling_sum_3d'] = input_data['Manual Daily Rainfall (mm)'] + input_data['rainfall_lag_1d'] + input_data['rainfall_lag_2d']
    input_data['rainfall_rolling_mean_7d'] = input_data['rainfall_rolling_mean_3d'] * 1.1 # Placeholder
    input_data['rainfall_rolling_sum_7d'] = input_data['rainfall_rolling_sum_3d'] * 1.5 # Placeholder
    input_data['rainfall_rolling_mean_14d'] = input_data['rainfall_rolling_mean_7d'] * 0.9 # Placeholder
    input_data['rainfall_rolling_sum_14d'] = input_data['rainfall_rolling_sum_7d'] * 1.2 # Placeholder

    input_data['neighbor_avg_rainfall_lag_1d_50km'] = manual_daily_rainfall * 0.9 # Example: nearby avg is 90% of current

    # --- One-Hot Encoding for Station, District, Tehsil ---
    # Initialize all one-hot encoded features to 0.0
    for feature in expected_features:
        if feature.startswith('Station_') or feature.startswith('District_') or feature.startswith('Tehsil_'):
            input_data[feature] = 0.0

    # Set the selected one-hot encoded features to 1.0
    input_data[f'Station_{selected_station}'] = 1.0
    input_data[f'District_{selected_district}'] = 1.0
    input_data[f'Tehsil_{selected_tehsil}'] = 1.0

    # Create DataFrame in the exact order of expected_features
    try:
        input_df = pd.DataFrame([input_data])[expected_features]
    except KeyError as e:
        st.error(f"Internal Error: Feature mismatch during DataFrame creation: {e}")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred during data preparation: {e}")
        st.stop()

    # --- Make Prediction ---
    st.subheader("Prediction Result")
    try:
        prediction = loaded_model.predict(input_df)[0]
        prediction_proba = loaded_model.predict_proba(input_df)[0]

        if prediction == 1:
            st.success("### ALERT: Extreme Rainfall Event Predicted! 🚨")
            st.markdown(f"Probability of Extreme Rainfall: **{prediction_proba[1]*100:.2f}%**")
            st.markdown(f"Probability of No Extreme Rainfall: {prediction_proba[0]*100:.2f}%")
        else:
            st.info("### No Extreme Rainfall Event Predicted.")
            st.markdown(f"Probability of No Extreme Rainfall: **{prediction_proba[0]*100:.2f}%**")
            st.markdown(f"Probability of Extreme Rainfall: {prediction_proba[1]*100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("Please check your input values and ensure the model file is correct.")

st.markdown("---")
st.markdown("Developed for Extreme Daily Rainfall Event Classification and Early Warning System.")