import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from meteostat import Point, Hourly
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="Toronto Traffic Forecast", layout="wide")

st.title("Toronto Traffic Flow Forecast (Traffic + Weather + XGBoost)")
st.write("Upload your traffic CSV, fetch hourly weather, train a model, and visualize forecasts.")

st.sidebar.header("Settings")

HOUR = st.sidebar.slider("Forecast horizon (hours ahead)", 1, 24, 3)

use_location = st.sidebar.checkbox("Filter by a specific location_name", value=False)
location_name = None
if use_location:
    location_name = st.sidebar.text_input("location_name (must match CSV)", value="")

weather_lat = st.sidebar.number_input("Weather latitude", value=43.6532, format="%.4f")
weather_lon = st.sidebar.number_input("Weather longitude", value=-79.3832, format="%.4f")

st.sidebar.markdown("---")
train_btn = st.sidebar.button("Train model")

uploaded = st.file_uploader("Upload traffic CSV (must include: time_start, volume_15min)", type=["csv"])

@st.cache_data(show_spinner=False)
def load_and_aggregate(csv_bytes, location_name=None):
    df = pd.read_csv(csv_bytes)
    df["time_start"] = pd.to_datetime(df["time_start"])

    if location_name and "location_name" in df.columns:
        df = df[df["location_name"] == location_name].copy()

    df["hour"] = df["time_start"].dt.floor("h")

    traffic_hourly = (
        df.groupby("hour")["volume_15min"]
          .sum()
          .reset_index()
          .rename(columns={"hour": "datetime", "volume_15min": "traffic_flow"})
          .sort_values("datetime")
          .reset_index(drop=True)
    )

    traffic_hourly["dt_diff"] = traffic_hourly["datetime"].diff().dt.total_seconds().div(3600)
    traffic_hourly = traffic_hourly[(traffic_hourly["dt_diff"].isna()) | (traffic_hourly["dt_diff"] <= 1)].drop(columns="dt_diff")
    traffic_hourly = traffic_hourly.sort_values("datetime").reset_index(drop=True)

    return traffic_hourly

@st.cache_data(show_spinner=False)
def fetch_weather(start, end, lat, lon):
    toronto = Point(lat, lon)
    weather = Hourly(toronto, start, end).fetch().reset_index()
    weather = weather.rename(columns={"time": "datetime"})
    wanted = ["datetime", "temp", "prcp", "wspd", "snow"]
    for c in wanted:
        if c not in weather.columns:
            weather[c] = 0.0
    weather = weather[wanted]
    return weather

def build_features(traffic_hourly, weather, horizon):
    data = pd.merge(traffic_hourly, weather, on="datetime", how="inner").copy()

    for col in ["temp", "prcp", "wspd", "snow"]:
        data[col] = data[col].fillna(0)

    data["hour"] = data["datetime"].dt.hour
    data["dayofweek"] = data["datetime"].dt.dayofweek
    data["is_weekend"] = (data["dayofweek"] >= 5).astype(int)

    for lag in [1, 2, 24]:
        data[f"lag_{lag}"] = data["traffic_flow"].shift(lag)

    data["target"] = data["traffic_flow"].shift(-horizon)
    data = data.dropna().reset_index(drop=True)

    data["baseline_pred"] = data["lag_1"]

    features = [
        "hour", "dayofweek", "is_weekend",
        "temp", "prcp", "wspd", "snow",
        "lag_1", "lag_2", "lag_24"
    ]
    return data, features

def train_xgb(data, features):
    split = int(len(data) * 0.8)
    train = data.iloc[:split].copy()
    test  = data.iloc[split:].copy()

    X_train, y_train = train[features], train["target"]
    X_test,  y_test  = test[features],  test["target"]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    test["prediction"] = model.predict(X_test)

    mae_model = mean_absolute_error(y_test, test["prediction"])
    mae_base  = mean_absolute_error(y_test, test["baseline_pred"])

    importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)

    return model, test, mae_base, mae_model, importance

if uploaded is None:
    st.info("Upload a CSV")
    st.stop()

with st.spinner("Handling traffic data"):
    traffic_hourly = load_and_aggregate(uploaded, location_name if use_location else None)

st.subheader("Traffic data")
st.write(f"Rows: {len(traffic_hourly):,}")
st.write(f"Date range: {traffic_hourly['datetime'].min()} → {traffic_hourly['datetime'].max()}")
st.dataframe(traffic_hourly.head(20), use_container_width=True)

start = traffic_hourly["datetime"].min()
end   = traffic_hourly["datetime"].max()

with st.spinner("Meteostat"):
    weather = fetch_weather(start, end, weather_lat, weather_lon)

st.subheader("Weather data")
st.write(f"Rows: {len(weather):,}")
st.dataframe(weather.head(20), use_container_width=True)

data, features = build_features(traffic_hourly, weather, HOUR)

st.subheader("Merged dataset")
st.write(f"Rows after merge: {len(data):,}")
st.dataframe(data.head(20), use_container_width=True)

csv_out = data.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download merged dataset (traffic_weather_hourly.csv)",
    data=csv_out,
    file_name="traffic_weather_hourly.csv",
    mime="text/csv"
)

if not train_btn:
    st.warning("Adjust settings in the sidebar, then click **Train model**.")
    st.stop()

with st.spinner("Training XGBoost model"):
    model, test, mae_base, mae_model, importance = train_xgb(data, features)

col1, col2 = st.columns(2)

with col1:
    st.metric("MAE (Baseline: lag_1)", f"{mae_base:,.2f}")
with col2:
    st.metric("MAE (XGBoost)", f"{mae_model:,.2f}")

st.subheader("Feature importance")
st.dataframe(importance.to_frame("importance"), use_container_width=True)

plot_df = test.tail(24 * 7).copy()

st.subheader("Forecast plot (last 7 days of test set)")
fig = plt.figure(figsize=(12, 5))
plt.plot(plot_df["datetime"], plot_df["traffic_flow"], label="Actual")
plt.plot(plot_df["datetime"], plot_df["prediction"], label="Predicted")
plt.plot(plot_df["datetime"], plot_df["baseline_pred"], label="Baseline")
plt.title("Hourly Traffic Flow Forecast")
plt.xlabel("Time")
plt.ylabel("Traffic Flow")
plt.legend()
plt.tight_layout()
st.pyplot(fig)

