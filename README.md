# deepET
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/singhamninder/etoapp/main/app.py)

This application makes estimations of reference evapotranspiration ($ET_o$) using deep learning models, using the data uploaded by the user with daily temperature as inputs. The application allows the user to select a model and provides predictions and visualizations based on the selected model. The code loads pre-trained models, prepares input features, makes predictions, and displays the results. It also provides a download button to download the results as a CSV file.
* The uploaded data should atleast have Date, Tmin, Tmax, Tav, columns to be able to access the temeprature based deep learning models.
    * Relative humidity [%] and  wind speed $[m s^{-1}]$, if available, can also be included as column names RH and U, respectively to get access to models trained using all five features.
* Enter Latitude and Longitude [decimal degrees].
* In addition to the $ET_o$ estimation from the selected model, the resulting file will also include estimations from the Hargreaves & Samani (HS) model.
