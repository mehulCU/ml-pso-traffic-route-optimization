import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def prepare_data():

    # -----------------------------------
    # 1. LOAD DATA
    # -----------------------------------

    df = pd.read_csv("traffic_all.csv")
    df["date"] = pd.to_datetime(df["date"])

    top_segments = pd.read_csv("ml/top_100_segments_train_only.csv")
    segment_ids = top_segments["segment_id"].tolist()

    df = df[df["segment_id"].isin(segment_ids)].copy()

    df = df.sort_values(
        ["segment_id", "date", "hour"]
    ).reset_index(drop=True)


    # -----------------------------------
    # 2. TIME FEATURES
    # -----------------------------------

    df["day_of_week"] = df["date"].dt.dayofweek

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype(int)

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )


    # -----------------------------------
    # 3. WEEKLY LAG
    # -----------------------------------

    # Traffic at the same road and
    # same hour 7 days earlier.

    df["lag_168"] = (
        df.groupby("segment_id")["probe_count"]
        .shift(168)
    )


    # -----------------------------------
    # 4. DATE SPLITS
    # -----------------------------------

    train_end = pd.Timestamp("2024-08-23")
    validation_start = pd.Timestamp("2024-08-24")
    test_start = pd.Timestamp("2024-08-26")

    train_mask = df["date"] <= train_end


    # -----------------------------------
    # 5. SCALE PROBE COUNT
    # -----------------------------------

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


    # -----------------------------------
    # 6. SCALE WEEKLY LAG
    # -----------------------------------

    lag_train_mask = (
        train_mask
        & df["lag_168"].notna()
    )

    lag_scaler = MinMaxScaler()

    df.loc[
        lag_train_mask,
        ["lag_168_scaled"]
    ] = lag_scaler.fit_transform(
        df.loc[
            lag_train_mask,
            ["lag_168"]
        ]
    )

    lag_valid_rows = df["lag_168"].notna()

    df.loc[
        lag_valid_rows & ~lag_train_mask,
        ["lag_168_scaled"]
    ] = lag_scaler.transform(
        df.loc[
            lag_valid_rows & ~lag_train_mask,
            ["lag_168"]
        ]
    )


    # -----------------------------------
    # 7. SCALE ROAD FEATURES
    # -----------------------------------

    road_features = [
        "speed_limit",
        "frc",
        "distance"
    ]

    df[road_features] = (
        df[road_features].astype(float)
    )

    road_scaler = MinMaxScaler()

    road_train = df.loc[
        train_mask,
        road_features
    ]

    road_scaler.fit(road_train)

    df.loc[
        train_mask,
        road_features
    ] = road_scaler.transform(
        road_train
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


    # -----------------------------------
    # 8. FINAL FEATURES
    # -----------------------------------

    features = [
        "probe_count_scaled",
        "hour_sin",
        "hour_cos",
        "day_of_week",
        "is_weekend",
        "speed_limit",
        "frc",
        "distance"
    ]

    time_steps = 24


    # -----------------------------------
    # 9. STORAGE
    # -----------------------------------

    X_train = []
    y_train = []

    X_validation = []
    y_validation = []

    X_test = []
    y_test = []

    weekly_lag_train = []
    weekly_lag_validation = []
    weekly_lag_test = []


    # -----------------------------------
    # 10. CREATE SEQUENCES
    # -----------------------------------

    for segment_id, road in df.groupby("segment_id"):

        road = road.sort_values(
            ["date", "hour"]
        ).reset_index(drop=True)

        values = road[features].values

        targets = (
            road["probe_count_scaled"].values
        )

        dates = road["date"].values

        weekly_lag = (
            road["lag_168_scaled"].values
        )


        for i in range(
            time_steps,
            len(road)
        ):

            target_date = pd.Timestamp(
                dates[i]
            )

            # We need the weekly reference
            # for the target hour.
            if np.isnan(weekly_lag[i]):
                continue

            sequence = values[
                i - time_steps:i
            ]

            target = targets[i]

            weekly_reference = weekly_lag[i]


            # -----------------------------------
            # TRAIN
            # -----------------------------------

            if target_date <= train_end:

                X_train.append(sequence)
                y_train.append(target)
                weekly_lag_train.append(
                    weekly_reference
                )


            # -----------------------------------
            # VALIDATION
            # -----------------------------------

            elif (
                target_date >= validation_start
                and target_date < test_start
            ):

                X_validation.append(sequence)
                y_validation.append(target)
                weekly_lag_validation.append(
                    weekly_reference
                )


            # -----------------------------------
            # TEST
            # -----------------------------------

            elif target_date >= test_start:

                X_test.append(sequence)
                y_test.append(target)
                weekly_lag_test.append(
                    weekly_reference
                )


    # -----------------------------------
    # 11. NUMPY ARRAYS
    # -----------------------------------

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_validation = np.array(X_validation)
    y_validation = np.array(y_validation)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    weekly_lag_train = np.array(
        weekly_lag_train
    )

    weekly_lag_validation = np.array(
        weekly_lag_validation
    )

    weekly_lag_test = np.array(
        weekly_lag_test
    )


    # -----------------------------------
    # 12. ADD WEEKLY LAG TO EVERY
    #     TIMESTEP AS A CONSTANT FEATURE
    # -----------------------------------

    # The weekly reference belongs to the
    # prediction target, not to each historical
    # hour. We therefore append it as one
    # additional feature repeated across the
    # 24-step sequence.

    weekly_train = np.repeat(
        weekly_lag_train[:, np.newaxis],
        time_steps,
        axis=1
    )

    weekly_validation = np.repeat(
        weekly_lag_validation[:, np.newaxis],
        time_steps,
        axis=1
    )

    weekly_test = np.repeat(
        weekly_lag_test[:, np.newaxis],
        time_steps,
        axis=1
    )


    weekly_train = weekly_train[:, :, np.newaxis]
    weekly_validation = weekly_validation[:, :, np.newaxis]
    weekly_test = weekly_test[:, :, np.newaxis]


    X_train = np.concatenate(
        [X_train, weekly_train],
        axis=2
    )

    X_validation = np.concatenate(
        [X_validation, weekly_validation],
        axis=2
    )

    X_test = np.concatenate(
        [X_test, weekly_test],
        axis=2
    )


    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        probe_scaler,
        lag_scaler,
        road_scaler
    )


# ===================================
# MAIN
# ===================================

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
        lag_scaler,
        road_scaler
    ) = data


    print("-----------------------------------")
    print("ENHANCED MULTI-ROAD LSTM DATA")
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

    print(
        "\nX_test:",
        X_test.shape
    )

    print(
        "y_test:",
        y_test.shape
    )

    print(
        "\nNumber of features:",
        X_train.shape[2]
    )

    print(
        "Time steps:",
        X_train.shape[1]
    )

    print("\nFirst timestep:")
    print(X_train[0][0])

    print("\nFirst sequence:")
    print(X_train[0])

    print("\nFirst target:")
    print(y_train[0])