import math
import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Traffic Route Optimization",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Dynamic Traffic Route Optimization")
st.caption("Streamlit frontend connected to your FastAPI ML backend")

with st.sidebar:
    st.header("Backend Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.ok:
            st.success("FastAPI connected")
        else:
            st.error(f"Health check: HTTP {health.status_code}")
    except requests.RequestException:
        st.error("FastAPI is not reachable. Start the backend on port 8000.")

st.header("1. Route Optimization")

c1, c2 = st.columns(2)
with c1:
    source = st.number_input("Source node ID", value=3319, step=1)
with c2:
    destination = st.number_input("Destination node ID", value=316, step=1)

timestamp = st.text_input(
    "Forecast timestamp",
    value="2024-08-26T10:00:00",
)

route_payload = {
    "source": int(source),
    "destination": int(destination),
    "forecast_timestamp": timestamp,
}

r1, r2 = st.columns(2)

with r1:
    if st.button("Find Forecast Route", use_container_width=True):
        try:
            with st.spinner("Calculating forecast route..."):
                response = requests.post(
                    f"{API_URL}/route",
                    json=route_payload,
                    timeout=120,
                )
            if response.ok:
                data = response.json()
                st.success("Forecast route calculated")
                st.json(data)
            else:
                st.error(f"API error: HTTP {response.status_code}")
                st.code(response.text)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

with r2:
    if st.button("Find PSO Route", use_container_width=True):
        try:
            with st.spinner("Running PSO..."):
                response = requests.post(
                    f"{API_URL}/route/pso",
                    json=route_payload,
                    timeout=120,
                )
            if response.ok:
                data = response.json()
                st.success("PSO route calculated")
                st.json(data)
            else:
                st.error(f"API error: HTTP {response.status_code}")
                st.code(response.text)
        except requests.RequestException as exc:
            st.error(f"Request failed: {exc}")

st.divider()
st.header("2. Probe Count Prediction")
st.write(
    "The API needs 24 chronological hourly observations. "
    "This form generates synthetic observations for testing the endpoint."
)

with st.form("prediction_form"):
    st.subheader("Road Features")

    a, b, c = st.columns(3)
    with a:
        starting_count = st.number_input(
            "Starting probe count", min_value=0.0, value=20.0
        )
        speed_limit = st.number_input(
            "Speed limit (km/h)", min_value=0.1, value=40.0
        )
    with b:
        frc = st.number_input("FRC", value=3.0)
        distance = st.number_input(
            "Road segment distance (meters)", min_value=0.1, value=100.0
        )
    with c:
        day_of_week = st.selectbox(
            "Day of week (Monday=0, Sunday=6)",
            options=list(range(7)),
            index=0,
        )
        hourly_change = st.number_input(
            "Synthetic hourly count change", value=0.0, step=1.0
        )

    st.caption(
        "Hour features are calculated from hours 0–23. "
        "The count changes by the selected amount each hour, never below zero."
    )
    submitted = st.form_submit_button(
        "Predict Next Hour", use_container_width=True
    )

if submitted:
    history = []
    for hour in range(24):
        history.append({
            "probe_count": max(0.0, starting_count + hour * hourly_change),
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "day_of_week": day_of_week,
            "speed_limit": speed_limit,
            "frc": frc,
            "distance": distance,
        })

    try:
        with st.spinner("Requesting prediction from FastAPI..."):
            response = requests.post(
                f"{API_URL}/predict",
                json={"history": history},
                timeout=120,
            )

        if response.ok:
            result = response.json()
            st.success("Prediction received")
            p1, p2, p3 = st.columns(3)
            p1.metric(
                "Predicted probe count",
                f"{result['predicted_probe_count']:.2f}",
            )
            p2.metric("Forecast horizon", result["forecast_horizon"])
            p3.metric("Observations used", result["observations_used"])
            st.json(result)
        else:
            st.error(f"API error: HTTP {response.status_code}")
            st.code(response.text)
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")

st.divider()
st.caption(
    "Note: probe count is not a direct measurement of traffic speed or "
    "travel time. Synthetic prediction inputs are for API testing only."
)
