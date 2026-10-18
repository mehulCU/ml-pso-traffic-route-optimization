# 🛣️ New Delhi Traffic Probe Count & Analytics Dataset (2024)

**Created by: Ryan Madhuwala (RAW), Founder of Garudex Labs**

## 📦 Overview

This dataset is a comprehensive, large-scale traffic analysis and urban infrastructure resource capturing **probe count data** and **urban metrics** across **New Delhi** and surrounding regions, featuring a **15,000+ km road network**.

It includes both **generalized annual traffic analytics for 2024** and **detailed probe count data from August 11–30, 2024**, a period rich in cultural and traffic variation due to Indian national festivals like **Independence Day, Rakshabandhan, and Janmashtami**, alongside the **monsoon season** — making this an invaluable dataset for mobility research, urban planning, AI traffic forecasting, smart city infrastructure, and congestion pattern modeling.

---

## 📁 Folder Structure

```bash
new_delhi_traffic_dataset/
│
├── facility/                  # Category-wise infrastructure features (education, healthcare, transport, etc.)
│   └── features_geopkg.json
│
├── geojson/                   # Administrative GeoJSON map of New Delhi
│   └── new_delhi.json
│
├── global_metrics/           # Aggregated city & urban traffic insights (speed, congestion, trends)
│   ├── 2024_city_rush_hour.json
│   ├── 2024_urban_rush_hour.json
│   ├── new_delhi_2024_city_traffic.json
│   └── new_delhi_2024_urban_traffic.json
│
├── probe_counts/geojson/     # Hourly probe count for each road (Aug 11–30, 2024)
│   ├── new_delhi__2024-08-11_to_2024-08-11_.geojson
│   ├── ...
│   └── new_delhi__2024-08-30_to_2024-08-30_.geojson
│
├── weekday_stats/            # Weekday-wise congestion, speed, and time metrics
│   ├── 2024_week_day_congestion_city.csv
│   ├── 2024_week_day_speed_city.csv
│   ├── 2024_week_day_time_city.csv
│   ├── 2024_week_day_congestion_urban.csv
│   ├── 2024_week_day_speed_urban.csv
│   └── 2024_week_day_time_urban.csv
│
└── README.md
```

---

## 📊 Dataset Highlights

### 🚦 **Hourly Road Probe Counts (Aug 11–30, 2024)**
- **Granular hourly data** for every road segment in Delhi NCR.
- Over **40 lakh entries per day** in GeoJSON format.
- Captures **festival effects**, **weekend vs weekday differences**, and **rainy season influence**.
- Ideal for **temporal trend analysis, anomaly detection, and event-driven traffic modeling**.

### 🌍 **Infrastructure Feature Classification**
- Roadside facility mapping in 10 well-defined categories:
  - Education, Healthcare, Retail, Government, Entertainment, Religious, Transport, Leisure, Mixed, Historic.
- Defined using **OpenStreetMap tags** and grouped for **urban analytics and planning** use cases.

### 📈 **Global Urban Traffic Metrics (2024)**
- **City-wide and Urban-level summaries**:
  - Average travel time, congestion levels, speed analysis.
  - Rush hour statistics with year-over-year comparisons vs 2023.
  - Worst travel days, peak congestion months.
  - Monthly congestion trends with 2023 reference.

```json
"morning_rush_hour": {
  "time_taken_10km": "25 min 28 s",
  "average_speed_kmh": 23.6,
  "congestion_level_percent": 43
}
```

---

## 📆 Why August 2024 Matters?
- Contains data from:
  - **Independence Day (Aug 15)**
  - **Rakshabandhan (Aug 19)**
  - **Janmashtami (Aug 26)**
- Monsoon rain season → Unique impact on traffic behavior and urban mobility patterns.

---

## 📌 Use Cases
- Smart city simulations
- Congestion forecasting models
- Road usage analysis
- Event-based anomaly detection
- Urban mobility infrastructure planning
- AI models for adaptive traffic signal systems

---

## 📬 Citation & Acknowledgement

If you use this dataset in your research or development, please cite:

**Ryan Madhuwala (RAW) _New Delhi Traffic Probe Count & Analytics Dataset (2024)._**

---

## 📫 Contact

**Ryan Madhuwala (RAW)**
Founder, Garudex Labs
📧 rawx18.dev@gmail.com
