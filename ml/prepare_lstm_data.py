import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_data():

    # 1. Load dataset
    df = pd.read_csv("traffic_all.csv")
    df["date"] = pd.to_datetime(df["date"])

    # 2. Select one road
    segment_id = -13560179880493
    df = df[df["segment_id"] == segment_id]
    df = df.sort_values(["date", "hour"]).reset_index(drop=True)

    # 3. Create time features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # 4. Define chronological split dates
    train_end = pd.Timestamp("2024-08-23")
    validation_start = pd.Timestamp("2024-08-24")
    test_start = pd.Timestamp("2024-08-26")

    # 5. Scale probe count using TRAINING data only
    scaler = MinMaxScaler()

    train_mask = df["date"] <= train_end

    df.loc[train_mask, "probe_count_scaled"] = scaler.fit_transform(
        df.loc[train_mask, ["probe_count"]]
    ).ravel()

    df.loc[~train_mask, "probe_count_scaled"] = scaler.transform(
        df.loc[~train_mask, ["probe_count"]]
    ).ravel()

    # 6. Features
    features = [
        "probe_count_scaled",
        "hour_sin",
        "hour_cos",
        "day_of_week"
    ]

    time_steps = 24

    # 7. Create sequences from the complete chronological data
    X = []
    y = []
    dates = []

    values = df[features].values
    targets = df["probe_count_scaled"].values

    for i in range(time_steps, len(df)):

        X.append(values[i - time_steps:i])
        y.append(targets[i])
        dates.append(df.iloc[i]["date"])

    X = np.array(X)
    y = np.array(y)
    dates = pd.to_datetime(dates)

    # 8. Split sequences based on the TARGET date
    train_mask = dates <= train_end

    validation_mask = (
        (dates >= validation_start) &
        (dates < test_start)
    )

    test_mask = dates >= test_start

    X_train = X[train_mask]
    y_train = y[train_mask]

    X_validation = X[validation_mask]
    y_validation = y[validation_mask]

    X_test = X[test_mask]
    y_test = y[test_mask]

    return (
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test,
        scaler
    )


if __name__ == "__main__":

    data = prepare_data()

    (
        X_train, y_train,
        X_validation, y_validation,
        X_test, y_test,
        scaler
    ) = data

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_validation:", X_validation.shape)
    print("y_validation:", y_validation.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print("\nFirst test target date:")
    print("Test starts at:", "2024-08-26")

    print("\nFirst test sequence:")
    print(X_test[0])

    print("\nFirst test target:")
    print(y_test[0])