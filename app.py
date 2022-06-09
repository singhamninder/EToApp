import streamlit as st
import pandas as pd
import numpy as np
import pyeto
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, AutoDateLocator
import pickle

def plot_results(df):
    fig, ax = plt.subplots()
    if 'Eto' in df.columns:
        ax.plot(df['Date'], df['Eto'], 'o', alpha=0.6, label = '$ET_{o}$')
        ax.plot(df['Date'], df['EThs'], 'o:', alpha=0.7, linewidth=2, label = '$ET_{HS}$')
        ax.plot(df['Date'], df[model], linewidth=2, ls ='--', label = model)
        ax.legend(prop={'size': 13})
        ax.grid(True,linestyle='--')
        ax.xaxis.set_major_formatter(DateFormatter('%d\n%b\n%Y'))
        ax.xaxis.set_major_locator(AutoDateLocator())
    else:
        ax.plot(df['Date'], df['EThs'], 'o:', alpha=0.7, linewidth=2, label = '$ET_{HS}$')
        ax.plot(df['Date'], df[model], linewidth=2, ls ='--', label = model)
        ax.legend(prop={'size': 13})
        ax.grid(True,linestyle='--')
        ax.xaxis.set_major_formatter(DateFormatter('%d\n%b\n%Y'))
        ax.xaxis.set_major_locator(AutoDateLocator())
    fig.supylabel('$ET{_{o}}$ $[mm d{^{-1}}$]', size = 16, weight='bold')
    return fig

def split_sequences(sequences, n_steps):
    '''Function to convert input timeseries to lag observations .i.e multiple samples with specified number of timesteps'''
    X= []
    for i in range(len(sequences)):
        end_ix = i + n_steps
        if end_ix > len(sequences):
            break
        seq_x = sequences[i:end_ix, :]
        X.append(seq_x)
    return np.array(X)

st.title('$ET_{o}$ Estimation using Machine Learning')
with st.sidebar:
    st.subheader('1. Upload your CSV file')
    uploaded_file = st.file_uploader("Make sure coloumns are named - Date, Tmin, Tmax, Tav, RH, and U", type=["csv"],
             help='File should atleast have columns - Date, Tmin, Tmax, and Tav')
    st.markdown('If available, you can also add reference evapotranspiration ($ET_{o}$) column. \
                Make sure to name it as Eto')

    st.subheader('2. Enter Latitude and Longitude [decimal degrees]')
    st.markdown('> 📝Do not leave these values to default, that will result in erroneous calculations')
    lat = st.number_input('Lat', value=33.964942, format ='%.3f')
    lon = st.number_input('Long', format ='%.3f', value=-117.33698)
    st.write('Current selection is ', lat, lon)
    location =pd.DataFrame(
    {'latitude': [lat],
     'longitude': [lon]})
    st.map(location, zoom=4)

st.subheader('Dataset')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['Date'] = pd.to_datetime(df['Date'])
    st.markdown('**Glimpse of your dataset**')
    st.write(df)
    if (set(['Tmin','Tmax', 'Tav','RH', 'U']).issubset(df.columns)) and (len(df)>30):
        st.success('All Required Columns are present')
        models = (['ANN_T', 'ANN_all', 'LSTM_T', 'LSTM_all', 'CNN_T', 'CNN_all'])
    elif set(['Tmin','Tmax', 'Tav']).issubset(df.columns) and (len(df)>30):
        st.success('Temperature Columns are present')
        models = (['ANN_T','LSTM_T','CNN_T'])
    elif (set(['Tmin','Tmax', 'Tav','RH', 'U']).issubset(df.columns)) and (len(df)<30):
        st.info('Need more than 30 rows to use LSTM and CNN-1D models')
        models = (['ANN_T', 'ANN_all'])
    elif set(['Tmin','Tmax', 'Tav']).issubset(df.columns) and (len(df)<30):
        st.info('Temperature Columns are present, Need more than 30 rows to use LSTM and CNN-1D models')
        models = (['ANN_T'])
    else:
        st.error('Please make sure required columns are present and are named correctly')
    st.write(f'Data has {df.shape[0]} rows and {df.shape[1]} columns')
    
    lat_r = pyeto.deg2rad(lat) # Convert latitude to radians
    doy = [d.timetuple().tm_yday for d in df.Date] # Day of the Year

    sol_dec = [pyeto.sol_dec(y) for y in doy]     #Solar declination
    ird = [pyeto.inv_rel_dist_earth_sun(y) for y in doy] #inverse relative distance Earth-Sun
    sha = [pyeto.sunset_hour_angle(lat_r,s) for s in sol_dec]  #sunset hour angle

    Ra = [] #Extraterrestrial radiation [MJ m-2 day-1] 
    for (i, j, k) in zip (sol_dec, sha, ird):
            Ra.append(pyeto.et_rad(lat_r, i, j, k))

    EThs = [] #ET predicted by Hargreaves and Samani eq, Reference evapotranspiration over grass (ETo) [mm day-1]
    for (aa, bb, cc, ra) in zip (df['Tmin'], df['Tmax'], df['Tav'], Ra):
        EThs.append(pyeto.hargreaves(aa, bb, cc, ra))

    st.header('Fit the Model')
    import tensorflow as tf
    model = st. selectbox('Select Model', models)
    if model == 'ANN_T':
        ANN_T = tf.keras.models.load_model('ANN_T.h5')
        scalerT = pickle.load(open('annT_scaler.pkl', 'rb'))
        #multiplied y 0.408 to convert Ra to mm/d
        features = pd.concat([df[['Tmin','Tmax', 'Tav']], pd.Series(Ra)*0.408], axis=1).rename({0:'Ra'}, axis=1).values
        features = scalerT.transform(features)
        y_pred = ANN_T.predict(features)
        st.markdown('**Results for your data**')
        if 'Eto' in df.columns:
            results=pd.concat([df['Date'], df['Eto'], pd.DataFrame(EThs, columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        else:
            results=pd.concat([df['Date'], pd.DataFrame(EThs, columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        fig = plot_results(results)
        st.pyplot(fig)
        st.markdown('$ET_{HS}$ = Hargreaves & Samani model')
        st.write(results)
        csv = results.to_csv(index=False)
        st.download_button("Download Results", csv, "file.csv",
                    "text/csv", key='download-csv')

    elif model == 'ANN_all':
        ANN_all = tf.keras.models.load_model('ANN_all.h5')
        scalerAll = pickle.load(open('annAll_scaler.pkl', 'rb'))
        #multiplied y 0.408 to convert Ra to mm/d
        features = pd.concat([df[['Tmin','Tmax', 'Tav','RH', 'U']], pd.Series(Ra)*0.408], axis=1).rename({0:'Ra'}, axis=1).values
        features = scalerAll.transform(features)
        y_pred = ANN_all.predict(features)
        st.markdown('**Results for your data**')
        if 'Eto' in df.columns:
            results=pd.concat([df['Date'], df['Eto'], pd.DataFrame(EThs, columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        else:
            results=pd.concat([df['Date'], pd.DataFrame(EThs, columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        fig = plot_results(results)
        st.pyplot(fig)
        st.markdown('$ET_{HS}$ = Hargreaves & Samani model')
        st.write(results)
        csv = results.to_csv(index=False)
        st.download_button("Download Results", csv, "file.csv",
                    "text/csv", key='download-csv')

    elif model == 'LSTM_T':
        LSTM_T = tf.keras.models.load_model('lstm_T.h5')
        scalerT = pickle.load(open('lstmT_scaler.pkl', 'rb'))
        #multiplied y 0.408 to convert Ra to mm/d
        features = pd.concat([df[['Tmin','Tmax', 'Tav']], pd.Series(Ra)*0.408], axis=1).rename({0:'Ra'}, axis=1).values
        features = split_sequences(features, n_steps=30)
        features = scalerT.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        y_pred = LSTM_T.predict(features)
        # st.write(y_pred)
        st.markdown('**Results for your data**')
        if 'Eto' in df.columns:
            results=pd.concat([df['Date'][29:].reset_index(drop=True), df['Eto'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        else:
            results=pd.concat([df['Date'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        fig = plot_results(results)
        st.pyplot(fig)
        st.markdown('$ET_{HS}$ = Hargreaves & Samani model')
        st.write(results)
        csv = results.to_csv(index=False)
        st.download_button("Download Results", csv, "file.csv",
                    "text/csv", key='download-csv')

    elif model == 'LSTM_all':
        LSTM_all = tf.keras.models.load_model('lstm_all.h5')
        scalerAll = pickle.load(open('lstmAll_scaler.pkl', 'rb'))
        #multiplied y 0.408 to convert Ra to mm/d
        features = pd.concat([df[['Tmin','Tmax', 'Tav','RH', 'U']], pd.Series(Ra)*0.408], axis=1).rename({0:'Ra'}, axis=1).values
        features = split_sequences(features, n_steps=30)
        features = scalerAll.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        y_pred = LSTM_all.predict(features)
        st.markdown('**Results for your data**')
        if 'Eto' in df.columns:
            results=pd.concat([df['Date'][29:].reset_index(drop=True), df['Eto'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        else:
            results=pd.concat([df['Date'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        fig = plot_results(results)
        st.pyplot(fig)
        st.markdown('$ET_{HS}$ = Hargreaves & Samani model')
        st.write(results)
        csv = results.to_csv(index=False)
        st.download_button("Download Results", csv, "file.csv",
                    "text/csv", key='download-csv')
    
    elif model == 'CNN_T':
        CNN_T = tf.keras.models.load_model('cnn_T.h5')
        scalerT = pickle.load(open('cnnT_scaler.pkl', 'rb'))
        #multiplied y 0.408 to convert Ra to mm/d
        features = pd.concat([df[['Tmin','Tmax', 'Tav']], pd.Series(Ra)*0.408], axis=1).rename({0:'Ra'}, axis=1).values
        features = split_sequences(features, n_steps=30)
        features = scalerT.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        y_pred = CNN_T.predict(features)
        # st.write(y_pred)
        st.markdown('**Results for your data**')
        if 'Eto' in df.columns:
            results=pd.concat([df['Date'][29:].reset_index(drop=True), df['Eto'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        else:
            results=pd.concat([df['Date'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        fig = plot_results(results)
        st.pyplot(fig)
        st.markdown('$ET_{HS}$ = Hargreaves & Samani model')
        st.write(results)
        csv = results.to_csv(index=False)
        st.download_button("Download Results", csv, "file.csv",
                    "text/csv", key='download-csv')

    elif model == 'CNN_all':
        CNN_all = tf.keras.models.load_model('cnn_all.h5')
        scalerAll = pickle.load(open('cnnAll_scaler.pkl', 'rb'))
        #multiplied y 0.408 to convert Ra to mm/d
        features = pd.concat([df[['Tmin','Tmax', 'Tav','RH', 'U']], pd.Series(Ra)*0.408], axis=1).rename({0:'Ra'}, axis=1).values
        features = split_sequences(features, n_steps=30)
        features = scalerAll.transform(features.reshape(-1, features.shape[-1])).reshape(features.shape)
        y_pred = CNN_all.predict(features)
        st.markdown('**Results for your data**')
        if 'Eto' in df.columns:
            results=pd.concat([df['Date'][29:].reset_index(drop=True), df['Eto'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        else:
            results=pd.concat([df['Date'][29:].reset_index(drop=True),
                    pd.DataFrame(EThs[29:], columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
        fig = plot_results(results)
        st.pyplot(fig)
        st.markdown('$ET_{HS}$ = Hargreaves & Samani model')
        st.write(results)
        csv = results.to_csv(index=False)
        st.download_button("Download Results", csv, "file.csv",
                    "text/csv", key='download-csv')
    
else:
    st.info('Awaiting for CSV file to be uploaded.')
    if st.button('Press to use Demo Data'):
        df = pd.read_csv('demo.csv')
        df['Date'] = pd.to_datetime(df['Date'])
        st.markdown('**Displaying the Demo dataset**')
        st.write(df)
        if set(['Tmin','Tmax', 'Tav','RH', 'U']).issubset(df.columns):
            st.success('All Required Columns are present')
            models = (['ANN_T', 'ANN_all'])
        elif set(['Tmin','Tmax', 'Tav']).issubset(df.columns):
            st.success('Temperature Columns are present')
            models = (['ANN_T'])
        else:
            st.error('Please make sure required columns are present and are named correctly')

        lat_r = pyeto.deg2rad(lat) # Convert latitude to radians
        doy = [d.timetuple().tm_yday for d in df.Date] # Day of the Year
        sol_dec = [pyeto.sol_dec(y) for y in doy]     #Solar declination
        ird = [pyeto.inv_rel_dist_earth_sun(y) for y in doy] #inverse relative distance Earth-Sun
        sha = [pyeto.sunset_hour_angle(lat_r,s) for s in sol_dec]  #sunset hour angle
        Ra = [] #Extraterrestrial radiation [MJ m-2 day-1] 
        for (i, j, k) in zip (sol_dec, sha, ird):
                Ra.append(pyeto.et_rad(lat_r, i, j, k))
        EThs = [] #ET predicted by Hargreaves and Samani eq, Reference evapotranspiration over grass (ETo) [mm day-1]
        for (aa, bb, cc, ra) in zip (df['Tmin'], df['Tmax'], df['Tav'], Ra):
            EThs.append(pyeto.hargreaves(aa, bb, cc, ra))

        st.header('Fit the Model')
        import tensorflow as tf
        model = st. selectbox('Select Model', models)
        if model == 'ANN_T':
            ANN_T = tf.keras.models.load_model('ANN_T.h5')
            scalerT = pickle.load(open('annT_scaler.pkl', 'rb'))
            #multiplied y 0.408 to convert Ra to mm/d
            features = pd.concat([df[['Tmin','Tmax', 'Tav']], pd.Series(Ra)*0.408], axis=1).rename({0:'Ra'}, axis=1).values
            features = scalerT.transform(features)
            y_pred = ANN_T.predict(features)
            st.markdown('**Results for demo data**')
            results=pd.concat([df['Date'], pd.DataFrame(EThs, columns=['EThs']), pd.DataFrame(y_pred, columns=[model])],axis=1)
            fig = plot_results(results)
            st.pyplot(fig)
            st.markdown('$ET_{HS}$ = Hargreaves & Samani model')
            st.write(results)
            csv = results.to_csv(index=False)
            st.download_button("Download Results", csv, "file.csv",
                        "text/csv", key='download-csv')