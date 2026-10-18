import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_data():

    df = pd.read_csv("traffic_all.csv")
    df["date"] = pd.to_datetime(df["date"])

    top_segments = pd.read_csv("ml/top_100_segments.csv")
    segment_ids = top_segments["segment_id"].tolist()

    df = df[df["segment_id"].isin(segment_ids)].copy()

    df = df.sort_values(
        ["segment_id", "date", "hour"]
    ).reset_index(drop=True)

    # Time features
    df["day_of_week"] = df["date"].dt.dayofweek

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    # Date split
    train_end = pd.Timestamp("2024-08-23")
    validation_start = pd.Timestamp("2024-08-24")
    test_start = pd.Timestamp("2024-08-26")

    train_mask = df["date"] <= train_end

    # Scale probe count
    probe_scaler = MinMaxScaler()

    df.loc[
        train_mask,
        "probe_count_scaled"
    ] = probe_scaler.fit_transform(
        df.loc[
            train_mask,
            ["probe_count"]
        ]
    ).ravel()

    df.loc[
        ~train_mask,
        "probe_count_scaled"
    ] = probe_scaler.transform(
        df.loc[
            ~train_mask,
            ["probe_count"]
        ]
    ).ravel()

    # Road features
    road_features = [
        "speed_limit",
        "frc",
        "distance"
    ]

    df[road_features] = (
        df[road_features].astype(float)
    )

    road_scaler = MinMaxScaler()

    df.loc[
        train_mask,
        road_features
    ] = road_scaler.fit_transform(
        df.loc[
            train_mask,
            road_features
        ]
    )

    df.loc[
        ~train_mask,
        road_features
    ] = road_scaler.transform(
        df.loc[
            ~train_mask,
            road_features
        ]
    )

    # 7 features
    features = [
        "probe_count_scaled",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "speed_limit",
        "frc",
        "distance"
    ]

    time_steps = 24

    X_train = []
    y_train = []

    X_validation = []
    y_validation = []

    X_test = []
    y_test = []

    # Build sequences independently for each road
    for segment_id, road in df.groupby("segment_id"):

        road = road.sort_values(
            ["date", "hour"]
        ).reset_index(drop=True)

        values = road[features].values
        targets = road["probe_count_scaled"].values
        dates = road["date"].values

        for i in range(
            time_steps,
            len(road)
        ):

            sequence = values[
                i - time_steps:i
            ]

            target = targets[i]

            target_date = pd.Timestamp(
                dates[i]
            )

            if target_date <= train_end:

                X_train.append(sequence)
                y_train.append(target)

            elif (
                target_date >= validation_start
                and target_date < test_start
            ):

                X_validation.append(sequence)
                y_validation.append(target)

            elif target_date >= test_start:

                X_test.append(sequence)
                y_test.append(target)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        probe_scaler,
        road_scaler
    )


if __name__ == "__main__":

    data = prepare_data()

    (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        probe_scaler,
        road_scaler
    ) = data

    print("-----------------------------------")
    print("7-FEATURE MULTI-ROAD LSTM DATA")
    print("-----------------------------------")

    print("\nX_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print(
        "\nX_validation:",
        X_validation.shape
    )

    print(
        "y_validation:",
        y_validation.shape
    )

    print("\nX_test:", X_test.shape)
    print("y_test:", y_test.shape)

    print(
        "\nNumber of features:",
        X_train.shape[2]
    )

    print(
        "Time steps:",
        X_train.shape[1]
    )