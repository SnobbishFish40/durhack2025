from analysis import DataPreprocessor, ProphetForecaster, ForecastVisualizer
import pandas as pd

# =============================================================================
# CONFIGURATION - Change these to analyze different properties
# =============================================================================

def run_analysis(filepath: str, forecast_periods: int = 365):
    SEASONALITY_MODE = 'additive'  # 'additive' or 'multiplicative'
    CHANGEPOINT_PRIOR = 0.05       # Higher = more flexible trend (0.001-0.5)

    header_end_line = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == '-END HEADER-':
                header_end_line = i + 1  # skip this line as well
                break
    columns = pd.read_csv(filepath, nrows=0, skiprows=header_end_line).columns.tolist()
    targets = columns[2:]  # List of columns to forecast: 'T2M', 'GWETTOP', 'PRECTOTCORR'
    #titles = ["Temperature Forecast"] #, "Surface Soil Wetness Forecast", "Precipitation Forecast"]

    for TARGET_COLUMN in targets:
        print(f"Processing {TARGET_COLUMN}...")

        preprocessor = DataPreprocessor(filepath)
        prophet_data = preprocessor.prepare_for_prophet(
            preprocessor.df, 
            target_column=TARGET_COLUMN, 
            year_column='YEAR', 
            doy_column='DOY'
        )

        forecaster = ProphetForecaster(
            seasonality_mode=SEASONALITY_MODE,
            changepoint_prior_scale=CHANGEPOINT_PRIOR
        )
        forecaster.create_model(
            yearly_seasonality=True,
            weekly_seasonality=False,
#    growth='flat'  # Flat trend - focuses on seasonality only
        )
        forecaster.fit(prophet_data)
        forecast = forecaster.predict(periods=forecast_periods, frequency='D', include_history=True)
        return forecast

#run_analysis('precipitation.csv', forecast_periods=365)

#visualizer = ForecastVisualizer()
#visualizer.plot_forecast(forecaster.model, forecast, historical_data=prophet_data, 
#                        title="Temperature Forecast")
#visualizer.plot_components(forecaster.model, forecast)
#visualizer.create_dashboard(forecaster.model, forecast, prophet_data)
