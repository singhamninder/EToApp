# deepET: Reference Evapotranspiration ($ET_o$) Estimation with Deep Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/singhamninder/etoapp/main/app.py)

**deepET** is a web application built with Streamlit that estimates daily reference evapotranspiration ($ET_o$). It allows users to upload their own meteorological data and uses a suite of pre-trained deep learning models (ANN, LSTM, CNN) to generate predictions.

## Features

- **Upload Your Data**: Bring your own meteorological data in a simple CSV format.
- **Multiple Models**: Choose from six different pre-trained deep learning models based on the data you provide.
- **Baseline Comparison**: Automatically calculates a baseline $ET_o$ using the conventional Hargreaves-Samani (HS) method for comparison.
- **Interactive Visualization**: View the results in an interactive plot showing your data (if provided), the HS model, and the deep learning model prediction.
- **Data Export**: Download the complete results, including your original data and all predictions, as a new CSV file.

## How to Use the Web App

1.  **Prepare Your Data**: Your data must be in a CSV file and contain the following columns with the exact names:
    -   `Date`: The date for the measurement (e.g., `YYYY-MM-DD`).
    -   `Tmin`: Daily minimum temperature (°C).
    -   `Tmax`: Daily maximum temperature (°C).
    -   `Tav`: Daily average temperature (°C).

2.  **(Optional) Add More Features**: To use the more advanced models, you can include:
    -   `RH`: Daily average relative humidity (%).
    -   `U`: Daily average wind speed (m/s).
    -   `Eto`: Your own measured or calculated reference $ET_o$ values for direct comparison in the plot.

3.  **Upload and Configure**:
    -   In the sidebar, upload your CSV file.
    -   Enter the correct **latitude** and **longitude** (in decimal degrees) for the location where the data was collected. This is crucial for accurate calculations.

4.  **Run and Download**:
    -   Select one of the available deep learning models from the dropdown menu. The available models will be determined automatically based on the columns in your uploaded file.
    -   The application will process the data and display the results.
    -   Click the "Download Results" button to save the predictions.

## Citation

If you use this application or the underlying models in your research, please cite the following publication:

> Singh, A., Haghverdi, A., 2023. Development and evaluation of temperature-based deep learning models to estimate reference evapotranspiration. *Artificial Intelligence in Agriculture*. [https://doi.org/10.1016/j.aiia.2023.08.003](https://doi.org/10.1016/j.aiia.2023.08.003)
