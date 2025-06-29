from utils import *

st.title("**deepET** - $ET_{o}$ Estimation using Deep Learning")
st.markdown(
    """This application makes estimations of reference evapotranspiration ($ET_o$) using deep learning models,
            using the data uploaded by the user with daily temperature as inputs.
            The application allows the user to select a model and provides predictions and visualizations based on the selected model.
            The code loads pre-trained models, prepares input features, makes predictions, and displays the results.
             It also provides a download button to download the results as a CSV file.
            """
)
st.markdown(
    "> **Units for Columns**: Tmin, Tmax, and Ta are in [$°C$], Relative Humidity, RH [%], daily average windspeed, U [$m s^{-1}$]."
)
with st.sidebar:
    st.subheader("1. Upload your CSV file")
    uploaded_file = st.file_uploader(
        "Make sure coloumns are named - Date, Tmin, Tmax, Tav, RH, and U",
        type=["csv"],
        help="File should atleast have columns - Date, Tmin, Tmax, and Tav",
    )
    st.markdown(
        "If available, you can also add reference evapotranspiration ($ET_{o}$) column. \
                Make sure to name it as Eto"
    )

    st.subheader("2. Enter Latitude and Longitude [decimal degrees]")
    st.markdown(
        "> 📝Do not leave these values to default, that will result in erroneous calculations"
    )
    lat = st.number_input("Lat", value=33.964942, format="%.3f")
    lon = st.number_input("Long", format="%.3f", value=-117.33698)
    st.write("Current selection is ", lat, lon)
    location = pd.DataFrame({"latitude": [lat], "longitude": [lon]})
    st.map(location, zoom=4)
    st.subheader("Citation")


def get_available_models(df):
    """Determines which models are available based on DataFrame columns and length."""
    has_temp_cols = set(["Tmin", "Tmax", "Tav"]).issubset(df.columns)
    has_all_cols = set(["Tmin", "Tmax", "Tav", "RH", "U"]).issubset(df.columns)
    is_long_enough = len(df) > 30

    if has_all_cols and is_long_enough:
        st.success("All Required Columns are present. All models are available.")
        return ["ANN_T", "ANN_all", "LSTM_T", "LSTM_all", "CNN_T", "CNN_all"]
    elif has_temp_cols and is_long_enough:
        st.success(
            "Temperature Columns are present. Temperature-based models are available."
        )
        return ["ANN_T", "LSTM_T", "CNN_T"]
    elif has_all_cols and not is_long_enough:
        st.info("Need more than 30 rows to use LSTM and CNN-1D models.")
        return ["ANN_T", "ANN_all"]
    elif has_temp_cols and not is_long_enough:
        st.info(
            "Temperature Columns are present, but need more than 30 rows to use LSTM and CNN-1D models."
        )
        return ["ANN_T"]
    else:
        st.error(
            "Please make sure required columns are present and are named correctly (Tmin, Tmax, Tav)."
        )
        return []


def process_prediction(df, model_name, ra_series, eths_series):
    """Loads a model, preprocesses data, makes predictions, and displays results."""
    import tensorflow as tf

    N_STEPS = 30
    MODEL_CONFIG = {
        "ANN_T": {
            "path": "model/ANN_T.h5",
            "scaler": "scaler/annT_scaler.pkl",
            "features": ["Tmin", "Tmax", "Tav"],
            "is_sequence": False,
        },
        "ANN_all": {
            "path": "model/ANN_all.h5",
            "scaler": "scaler/annAll_scaler.pkl",
            "features": ["Tmin", "Tmax", "Tav", "RH", "U"],
            "is_sequence": False,
        },
        "LSTM_T": {
            "path": "model/lstm_T.h5",
            "scaler": "scaler/dlT_scaler.pkl",
            "features": ["Tmin", "Tmax", "Tav"],
            "is_sequence": True,
        },
        "LSTM_all": {
            "path": "model/lstm_all.h5",
            "scaler": "scaler/dlAll_scaler.pkl",
            "features": ["Tmin", "Tmax", "Tav", "RH", "U"],
            "is_sequence": True,
        },
        "CNN_T": {
            "path": "model/cnn_T.h5",
            "scaler": "scaler/dlT_scaler.pkl",
            "features": ["Tmin", "Tmax", "Tav"],
            "is_sequence": True,
        },
        "CNN_all": {
            "path": "model/cnn_all.h5",
            "scaler": "scaler/dlAll_scaler.pkl",
            "features": ["Tmin", "Tmax", "Tav", "RH", "U"],
            "is_sequence": True,
        },
    }
    config = MODEL_CONFIG[model_name]
    model = tf.keras.models.load_model(config["path"])
    scaler = pickle.load(open(config["scaler"], "rb"))

    features = (
        pd.concat([df[config["features"]], pd.Series(ra_series) * 0.408], axis=1)
        .rename({0: "Ra"}, axis=1)
        .values
    )

    if config["is_sequence"]:
        features = split_sequences(features, n_steps=N_STEPS)
        features = scaler.transform(features.reshape(-1, features.shape[-1])).reshape(
            features.shape
        )
    else:
        features = scaler.transform(features)

    y_pred = model.predict(features)

    start_idx = N_STEPS - 1 if config["is_sequence"] else 0
    results_parts = [df["Date"][start_idx:].reset_index(drop=True)]
    if "Eto" in df.columns:
        results_parts.append(df["Eto"][start_idx:].reset_index(drop=True))
    results_parts.extend(
        [
            pd.DataFrame(eths_series[start_idx:], columns=["EThs"]).reset_index(
                drop=True
            ),
            pd.DataFrame(y_pred, columns=[model_name]),
        ]
    )
    return pd.concat(results_parts, axis=1)


st.subheader("Dataset")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df["Date"] = pd.to_datetime(df["Date"])
    st.markdown("**Glimpse of your dataset**")
    st.write(df)
    models = get_available_models(df)
    st.write(f"Data has {df.shape[0]} rows and {df.shape[1]} columns")

    lat_r = pyeto.deg2rad(lat)  # Convert latitude to radians
    doy = [d.timetuple().tm_yday for d in df.Date]  # Day of the Year

    sol_dec = [pyeto.sol_dec(y) for y in doy]  # Solar declination
    ird = [
        pyeto.inv_rel_dist_earth_sun(y) for y in doy
    ]  # inverse relative distance Earth-Sun
    sha = [pyeto.sunset_hour_angle(lat_r, s) for s in sol_dec]  # sunset hour angle

    Ra = []  # Extraterrestrial radiation [MJ m-2 day-1]
    for i, j, k in zip(sol_dec, sha, ird):
        Ra.append(pyeto.et_rad(lat_r, i, j, k))

    EThs = (
        []
    )  # ET predicted by Hargreaves and Samani eq, Reference evapotranspiration over grass (ETo) [mm day-1]
    for aa, bb, cc, ra in zip(df["Tmin"], df["Tmax"], df["Tav"], Ra):
        EThs.append(pyeto.hargreaves(aa, bb, cc, ra))

    if models:
        st.header("Fit the Model")
        model_name = st.selectbox("Select Model", models)
        if model_name:
            with st.spinner(f"Running model {model_name}..."):
                results = process_prediction(df, model_name, Ra, EThs)

            st.markdown("**Results for your data**")
            fig = plot_results(results, model_name)
            st.pyplot(fig)
            st.markdown("$ET_{HS}$ = Hargreaves & Samani model")
            st.write(results)
            csv = results.to_csv(index=False)
            st.download_button(
                "Download Results",
                csv,
                f"{model_name}_results.csv",
                "text/csv",
                key=f"download-csv-{model_name}",
            )

else:
    st.info("Awaiting for CSV file to be uploaded.")
    if st.button("Press to use Demo Data"):
        df = pd.read_csv("demo.csv")
        df["Date"] = pd.to_datetime(df["Date"])
        st.markdown("**Displaying the Demo dataset**")
        st.write(df)
        if set(["Tmin", "Tmax", "Tav", "RH", "U"]).issubset(df.columns):
            st.success("All Required Columns are present")
            models = ["ANN_T", "ANN_all"]
        elif set(["Tmin", "Tmax", "Tav"]).issubset(df.columns):
            st.success("Temperature Columns are present")
            models = ["ANN_T"]
        else:
            st.error(
                "Please make sure required columns are present and are named correctly"
            )

        lat_r = pyeto.deg2rad(lat)  # Convert latitude to radians
        doy = [d.timetuple().tm_yday for d in df.Date]  # Day of the Year
        sol_dec = [pyeto.sol_dec(y) for y in doy]  # Solar declination
        ird = [
            pyeto.inv_rel_dist_earth_sun(y) for y in doy
        ]  # inverse relative distance Earth-Sun
        sha = [pyeto.sunset_hour_angle(lat_r, s) for s in sol_dec]  # sunset hour angle
        Ra = []  # Extraterrestrial radiation [MJ m-2 day-1]
        for i, j, k in zip(sol_dec, sha, ird):
            Ra.append(pyeto.et_rad(lat_r, i, j, k))
        EThs = (
            []
        )  # ET predicted by Hargreaves and Samani eq, Reference evapotranspiration over grass (ETo) [mm day-1]
        for aa, bb, cc, ra in zip(df["Tmin"], df["Tmax"], df["Tav"], Ra):
            EThs.append(pyeto.hargreaves(aa, bb, cc, ra))

        st.header("Fit the Model")
        import tensorflow as tf

        model = st.selectbox("Select Model", models)
        if model == "ANN_T":
            ANN_T = tf.keras.models.load_model("model/ANN_T.h5")
            scalerT = pickle.load(open("scaler/annT_scaler.pkl", "rb"))
            # multiplied y 0.408 to convert Ra to mm/d
            features = (
                pd.concat([df[["Tmin", "Tmax", "Tav"]], pd.Series(Ra) * 0.408], axis=1)
                .rename({0: "Ra"}, axis=1)
                .values
            )
            features = scalerT.transform(features)
            y_pred = ANN_T.predict(features)
            st.markdown("**Results for demo data**")
            results = pd.concat(
                [
                    df["Date"],
                    pd.DataFrame(EThs, columns=["EThs"]),
                    pd.DataFrame(y_pred, columns=[model]),
                ],
                axis=1,
            )

            fig = plot_results(results, model)
            st.pyplot(fig)
            st.markdown("$ET_{HS}$ = Hargreaves & Samani model")
            st.write(results)
            csv = results.to_csv(index=False)
            st.download_button(
                "Download Results", csv, "file.csv", "text/csv", key="download-csv"
            )

st.subheader("Citation")
st.markdown(
    """
        If you use this application in your research, please cite:
        
        Singh, A., Haghverdi, A., 2023. Development and evaluation of temperature-based deep learning models to estimate reference evapotranspiration. *Artificial Intelligence in Agriculture*. [https://doi.org/10.1016/j.aiia.2023.08.003](https://doi.org/10.1016/j.aiia.2023.08.003)
        """
)
