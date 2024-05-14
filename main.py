import streamlit as st
import pandas as pd
from datetime import datetime
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go

# Read CSV file
df = pd.read_csv('filled_data.csv', parse_dates=['Date'], date_format='%d-%m-%Y')
df.columns = ['ds', 'y', 'Rel Humidity at 2 Meters (%)', 'Precipitation Corrected (mm/day)', 'Surface Soil Wetness', 'Root Zone Soil Wetness', 'Temperature at 2 Meters', 'T2M_MAX', 'T2M_MIN']

# Get the minimum and maximum dates from the CSV file
min_date = df['ds'].min().date()
max_date = df['ds'].max().date()

st.title('Prediction Graph of Prices App')

# Get the user-selected date range
start_date = st.date_input("Select start date", min_date)
end_date = st.date_input("Select end date", max_date, min_value=start_date)

# Filter the data based on the selected date range
filtered_data = df[(df['ds'].dt.date >= start_date) & (df['ds'].dt.date <= end_date)]

# Prophet model
df_train = filtered_data[['ds', 'y']]
m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=0)  # No future periods
forecast = m.predict(future)

# Show and plot forecast
st.subheader('Prediction Graph data')
st.write(forecast[['ds', 'yhat']].tail())
st.write(f'Prediction price plot for the selected date range')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Predicted components")
fig2 = m.plot_components(forecast)
st.write(fig2)