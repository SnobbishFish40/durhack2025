"""
NASA Data Analysis and Prediction Pipeline using Prophet
=========================================================

This module provides functionality for:
1. Data preprocessing and cleaning
2. Time series forecasting using Facebook Prophet
3. Visualization and evaluation of predictions

NOTE: Data fetching from NASA APIs is handled by another team member.
      This module expects to receive preprocessed data as pandas DataFrames.

Author: DurHack 2025 Team
Date: November 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

# Uncomment when ready to use:
# from prophet import Prophet
# import matplotlib.pyplot as plt
# import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 2: DATA PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """
    Handles data cleaning, transformation, and preparation for Prophet.
    
    Prophet requires data in specific format:
    - Column 'ds' for dates (datetime)
    - Column 'y' for the target variable (numeric)
    - Optional: additional regressors as separate columns
    """
    
    def __init__(self):
        self.scaler = None  # For normalization if needed
        self.original_columns = None  # Store original column names
        
    def prepare_for_prophet(self, df: pd.DataFrame, 
                           date_column: str, 
                           target_column: str,
                           handle_missing: bool = True) -> pd.DataFrame:
        """
        Transform raw data into Prophet-compatible format.
        
        Args:
            df: Input DataFrame
            date_column: Name of the date/time column
            target_column: Name of the target variable column
            handle_missing: Whether to automatically handle missing values
            
        Returns:
            DataFrame with 'ds' and 'y' columns, sorted and cleaned
        """
        logger.info(f"Preparing data for Prophet (input shape: {df.shape})")
        
        # Store original columns for reference
        self.original_columns = df.columns.tolist()
        
        # Create a copy to avoid modifying original
        result = df.copy()
        
        # 1. Convert date column to datetime
        if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
            logger.info(f"Converting {date_column} to datetime")
            result[date_column] = pd.to_datetime(result[date_column], errors='coerce')
        
        # 2. Check for invalid dates
        invalid_dates = result[date_column].isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Found {invalid_dates} invalid dates, removing them")
            result = result.dropna(subset=[date_column])
        
        # 3. Ensure target column is numeric
        if not pd.api.types.is_numeric_dtype(result[target_column]):
            logger.info(f"Converting {target_column} to numeric")
            result[target_column] = pd.to_numeric(result[target_column], errors='coerce')
        
        # 4. Create Prophet format with 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': result[date_column],
            'y': result[target_column]
        })
        
        # 5. Handle missing values in target
        missing_count = prophet_df['y'].isna().sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in target column")
            if handle_missing:
                prophet_df = self.handle_missing_values(prophet_df, method='interpolate')
            else:
                logger.info("Dropping rows with missing target values")
                prophet_df = prophet_df.dropna(subset=['y'])
        
        # 6. Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # 7. Remove duplicates (keep first occurrence)
        duplicates = prophet_df.duplicated(subset=['ds']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate dates, keeping first occurrence")
            prophet_df = prophet_df.drop_duplicates(subset=['ds'], keep='first')
        
        # 8. Check for reasonable data
        if len(prophet_df) < 2:
            raise ValueError(f"Insufficient data after preprocessing: only {len(prophet_df)} rows")
        
        logger.info(f"Data prepared for Prophet (output shape: {prophet_df.shape})")
        logger.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        logger.info(f"Target range: {prophet_df['y'].min():.2f} to {prophet_df['y'].max():.2f}")
        
        return prophet_df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             method: str = 'interpolate',
                             column: str = 'y') -> pd.DataFrame:
        """
        Handle missing values in time series data.
        
        Args:
            df: Input DataFrame
            method: 'interpolate', 'forward_fill', 'backward_fill', 'mean', 'drop'
            column: Column to handle missing values for (default: 'y')
            
        Returns:
            DataFrame with missing values handled
        """
        result = df.copy()
        missing_before = result[column].isna().sum()
        
        if missing_before == 0:
            return result
        
        logger.info(f"Handling {missing_before} missing values using method: {method}")
        
        if method == 'interpolate':
            # Linear interpolation for time series
            result[column] = result[column].interpolate(method='linear', limit_direction='both')
        elif method == 'forward_fill':
            result[column] = result[column].fillna(method='ffill')
        elif method == 'backward_fill':
            result[column] = result[column].fillna(method='bfill')
        elif method == 'mean':
            result[column] = result[column].fillna(result[column].mean())
        elif method == 'drop':
            result = result.dropna(subset=[column])
        else:
            raise ValueError(f"Unknown method: {method}")
        
        missing_after = result[column].isna().sum()
        logger.info(f"Missing values after handling: {missing_after}")
        
        return result
    
    def detect_outliers(self, df: pd.DataFrame, 
                       column: str = 'y',
                       method: str = 'iqr',
                       threshold: float = 1.5,
                       action: str = 'flag') -> pd.DataFrame:
        """
        Detect and optionally remove outliers.
        
        Args:
            df: Input DataFrame
            column: Column to check for outliers (default: 'y')
            method: 'iqr' (Interquartile Range) or 'zscore'
            threshold: Multiplier for outlier detection (IQR: 1.5 standard, Z-score: 3.0 standard)
            action: 'flag' (add column), 'remove' (filter out), or 'cap' (cap at bounds)
            
        Returns:
            DataFrame with outliers handled based on action
        """
        result = df.copy()
        
        if method == 'iqr':
            # Interquartile Range method
            Q1 = result[column].quantile(0.25)
            Q3 = result[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (result[column] < lower_bound) | (result[column] > upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mean = result[column].mean()
            std = result[column].std()
            z_scores = np.abs((result[column] - mean) / std)
            outlier_mask = z_scores > threshold
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            
        else:
            raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'zscore'")
        
        outlier_count = outlier_mask.sum()
        logger.info(f"Detected {outlier_count} outliers using {method} method (threshold={threshold})")
        
        if outlier_count > 0:
            if action == 'flag':
                result['is_outlier'] = outlier_mask
            elif action == 'remove':
                result = result[~outlier_mask].reset_index(drop=True)
                logger.info(f"Removed {outlier_count} outliers")
            elif action == 'cap':
                result[column] = result[column].clip(lower=lower_bound, upper=upper_bound)
                logger.info(f"Capped {outlier_count} outliers to bounds [{lower_bound:.2f}, {upper_bound:.2f}]")
            else:
                raise ValueError(f"Unknown action: {action}. Use 'flag', 'remove', or 'cap'")
        
        return result
    
    def add_external_regressors(self, df: pd.DataFrame, 
                               regressors: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Add external variables that might influence predictions.
        
        Examples for NASA data:
        - Solar activity indices
        - Orbital parameters
        - Seasonal indicators
        
        Args:
            df: Prophet-formatted DataFrame (must have 'ds' column)
            regressors: Dictionary of regressor name -> series
            
        Returns:
            DataFrame with additional regressor columns
        """
        result = df.copy()
        
        for name, series in regressors.items():
            if len(series) != len(df):
                raise ValueError(f"Regressor '{name}' length ({len(series)}) doesn't match data length ({len(df)})")
            
            result[name] = series.values
            logger.info(f"Added regressor: {name}")
        
        return result
    
    def resample_timeseries(self, df: pd.DataFrame, 
                           frequency: str = 'D',
                           aggregation: str = 'mean',
                           date_column: str = 'ds',
                           value_column: str = 'y') -> pd.DataFrame:
        """
        Resample time series to different frequency.
        
        Args:
            df: Input DataFrame with datetime column
            frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'H' (hourly)
            aggregation: 'mean', 'sum', 'median', 'max', 'min', 'first', 'last'
            date_column: Name of date column (default: 'ds')
            value_column: Name of value column to aggregate (default: 'y')
            
        Returns:
            Resampled DataFrame
        """
        result = df.copy()
        
        # Ensure date column is datetime
        if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
            result[date_column] = pd.to_datetime(result[date_column])
        
        # Set date as index for resampling
        result = result.set_index(date_column)
        
        # Perform resampling based on aggregation method
        agg_functions = {
            'mean': 'mean',
            'sum': 'sum',
            'median': 'median',
            'max': 'max',
            'min': 'min',
            'first': 'first',
            'last': 'last'
        }
        
        if aggregation not in agg_functions:
            raise ValueError(f"Unknown aggregation: {aggregation}. Use one of {list(agg_functions.keys())}")
        
        resampled = result[value_column].resample(frequency).agg(agg_functions[aggregation])
        
        # Convert back to DataFrame with proper columns
        result_df = pd.DataFrame({
            date_column: resampled.index,
            value_column: resampled.values
        }).reset_index(drop=True)
        
        # Remove any NaN values that might have been created
        result_df = result_df.dropna()
        
        logger.info(f"Resampled from {len(df)} to {len(result_df)} rows (frequency: {frequency}, aggregation: {aggregation})")
        
        return result_df
    
    def create_lag_features(self, df: pd.DataFrame, 
                           column: str = 'y',
                           lags: List[int] = [1, 7, 30]) -> pd.DataFrame:
        """
        Create lagged features for time series analysis.
        Can be used as additional regressors in Prophet.
        
        Args:
            df: Input DataFrame
            column: Column to create lags for (default: 'y')
            lags: List of lag periods (e.g., [1, 7, 30] for 1-day, 1-week, 1-month)
            
        Returns:
            DataFrame with additional lag columns
        """
        result = df.copy()
        
        for lag in lags:
            lag_col_name = f'{column}_lag_{lag}'
            result[lag_col_name] = result[column].shift(lag)
            logger.info(f"Created lag feature: {lag_col_name}")
        
        # Note: First rows will have NaN for lag features
        nan_count = result[f'{column}_lag_{max(lags)}'].isna().sum()
        logger.warning(f"Lag features created {nan_count} NaN values in first rows")
        
        return result
    
    def aggregate_by_location(self, df: pd.DataFrame,
                             date_column: str,
                             value_column: str,
                             lat_column: str = 'latitude',
                             lon_column: str = 'longitude',
                             aggregation: str = 'mean') -> pd.DataFrame:
        """
        Aggregate multiple locations to a single time series.
        Useful when you have spatial data and want a regional average.
        
        Args:
            df: Input DataFrame with location and value data
            date_column: Name of date column
            value_column: Name of value column to aggregate
            lat_column: Name of latitude column
            lon_column: Name of longitude column
            aggregation: 'mean', 'median', 'sum', 'max', 'min'
            
        Returns:
            DataFrame with aggregated values by date
        """
        logger.info(f"Aggregating {len(df)} rows by date using {aggregation}")
        
        # Group by date and aggregate
        agg_dict = {value_column: aggregation}
        result = df.groupby(date_column).agg(agg_dict).reset_index()
        
        logger.info(f"Aggregated to {len(result)} unique dates")
        
        return result


# =============================================================================
# SECTION 3: PROPHET MODEL CONFIGURATION & TRAINING
# =============================================================================

class ProphetForecaster:
    """
    Wrapper for Facebook Prophet with NASA-specific configurations.
    """
    
    def __init__(self, seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05):
        """
        Initialize Prophet model with configuration.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
            changepoint_prior_scale: Flexibility of trend changes (0.001-0.5)
        """
        self.model = None
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.fitted = False
        
    def create_model(self, **kwargs) -> None:
        """
        Create and configure Prophet model.
        
        Common configurations for NASA data:
        - yearly_seasonality: For annual patterns (e.g., orbital cycles)
        - weekly_seasonality: For weekly patterns
        - daily_seasonality: For daily patterns
        - holidays: Special events (solar eclipses, meteor showers, etc.)
        """
        # TODO: Implement model creation
        # self.model = Prophet(
        #     seasonality_mode=self.seasonality_mode,
        #     changepoint_prior_scale=self.changepoint_prior_scale,
        #     **kwargs
        # )
        pass
    
    def add_custom_seasonality(self, name: str, period: float, 
                              fourier_order: int) -> None:
        """
        Add custom seasonality patterns.
        
        Examples for NASA data:
        - Lunar cycle: period=29.53 days
        - Solar cycle: period=11 years (4018 days)
        - Orbital periods: varies by planet
        
        Args:
            name: Name of the seasonality
            period: Period in days
            fourier_order: Number of Fourier terms (complexity)
        """
        # TODO: Implement custom seasonality
        pass
    
    def add_holidays(self, holidays_df: pd.DataFrame) -> None:
        """
        Add special events/holidays to the model.
        
        Args:
            holidays_df: DataFrame with 'holiday' and 'ds' columns
                        Optionally include 'lower_window' and 'upper_window'
        """
        # TODO: Implement holiday addition
        pass
    
    def fit(self, df: pd.DataFrame) -> None:
        """
        Train the Prophet model on historical data.
        
        Args:
            df: Prophet-formatted DataFrame (ds, y columns)
        """
        # TODO: Implement model fitting
        logger.info("Training Prophet model...")
        # self.model.fit(df)
        self.fitted = True
        pass
    
    def predict(self, periods: int, frequency: str = 'D') -> pd.DataFrame:
        """
        Generate future predictions.
        
        Args:
            periods: Number of periods to forecast
            frequency: 'D' (daily), 'W' (weekly), 'M' (monthly), 'H' (hourly)
            
        Returns:
            DataFrame with predictions, including yhat, yhat_lower, yhat_upper
        """
        # TODO: Implement prediction
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        pass
    
    def cross_validate(self, df: pd.DataFrame, 
                      initial: str, period: str, horizon: str) -> pd.DataFrame:
        """
        Perform time series cross-validation.
        
        Args:
            df: Prophet-formatted DataFrame
            initial: Initial training period (e.g., '730 days')
            period: Period between cutoff dates (e.g., '180 days')
            horizon: Forecast horizon (e.g., '365 days')
            
        Returns:
            DataFrame with cross-validation results
        """
        # TODO: Implement cross-validation
        # from prophet.diagnostics import cross_validation
        pass
    
    def calculate_metrics(self, cv_results: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics.
        
        Returns:
            Dictionary with MAE, RMSE, MAPE, coverage
        """
        # TODO: Implement metrics calculation
        # from prophet.diagnostics import performance_metrics
        pass


# =============================================================================
# SECTION 4: VISUALIZATION
# =============================================================================

class ForecastVisualizer:
    """
    Create visualizations for Prophet forecasts and NASA data analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid'):
        """
        Initialize visualizer with plotting style.
        """
        # plt.style.use(style)
        pass
    
    def plot_forecast(self, model, forecast: pd.DataFrame, 
                     historical_data: pd.DataFrame = None,
                     title: str = "NASA Data Forecast") -> None:
        """
        Plot forecast with confidence intervals.
        
        Args:
            model: Fitted Prophet model
            forecast: Forecast DataFrame from Prophet
            historical_data: Optional historical data to overlay
            title: Plot title
        """
        # TODO: Implement forecast plotting
        # model.plot(forecast)
        # plt.title(title)
        # plt.xlabel('Date')
        # plt.ylabel('Value')
        pass
    
    def plot_components(self, model, forecast: pd.DataFrame) -> None:
        """
        Plot forecast components (trend, seasonality, holidays).
        
        Args:
            model: Fitted Prophet model
            forecast: Forecast DataFrame
        """
        # TODO: Implement component plotting
        # model.plot_components(forecast)
        pass
    
    def plot_cross_validation(self, cv_results: pd.DataFrame) -> None:
        """
        Visualize cross-validation results.
        
        Args:
            cv_results: Cross-validation results from Prophet
        """
        # TODO: Implement CV visualization
        pass
    
    def create_dashboard(self, data: Dict[str, pd.DataFrame]) -> None:
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            data: Dictionary containing various DataFrames to visualize
        """
        # TODO: Implement dashboard creation
        # Consider using subplots or plotly for interactive dashboards
        pass


# =============================================================================
# SECTION 5: MAIN PIPELINE
# =============================================================================

class NASAForecastPipeline:
    """
    End-to-end pipeline for NASA data analysis and forecasting.
    Receives data from other team members (no data fetching).
    """
    
    def __init__(self):
        """
        Initialize the complete pipeline.
        """
        self.loader = DataLoader()
        self.preprocessor = DataPreprocessor()
        self.forecaster = ProphetForecaster()
        self.visualizer = ForecastVisualizer()
        
    def run_pipeline(self, data: pd.DataFrame,
                    date_column: str,
                    target_column: str,
                    forecast_periods: int = 365,
                    frequency: str = 'D',
                    **model_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete analysis pipeline on provided data.
        
        Args:
            data: Input DataFrame from data fetching team
            date_column: Name of date/time column in input data
            target_column: Name of target variable column
            forecast_periods: Number of periods to forecast
            frequency: Forecast frequency ('D', 'W', 'M', 'H')
            **model_kwargs: Additional Prophet model parameters
            
        Returns:
            Tuple of (processed_historical_data, forecast_data)
        """
        logger.info("Starting NASA forecast pipeline")
        
        # Step 1: Validate input data
        logger.info("Validating input data...")
        self.loader.validate_data(data, required_columns=[date_column, target_column])
        
        # Step 2: Preprocess
        logger.info("Preprocessing data...")
        processed_data = self.preprocessor.prepare_for_prophet(
            data, date_column, target_column
        )
        
        # Step 3: Train model
        logger.info("Training Prophet model...")
        self.forecaster.create_model(**model_kwargs)
        self.forecaster.fit(processed_data)
        
        # Step 4: Generate forecast
        logger.info(f"Generating {forecast_periods}-period forecast...")
        forecast = self.forecaster.predict(periods=forecast_periods, frequency=frequency)
        
        # Step 5: Visualize
        logger.info("Creating visualizations...")
        self.visualizer.plot_forecast(
            self.forecaster.model, 
            forecast, 
            processed_data
        )
        self.visualizer.plot_components(self.forecaster.model, forecast)
        
        logger.info("Pipeline complete!")
        return processed_data, forecast
    
    def run_with_cross_validation(self, data: pd.DataFrame,
                                  date_column: str,
                                  target_column: str,
                                  cv_initial: str = '730 days',
                                  cv_period: str = '180 days',
                                  cv_horizon: str = '365 days',
                                  **model_kwargs) -> Dict[str, float]:
        """
        Run pipeline with cross-validation to evaluate forecast accuracy.
        
        Args:
            data: Input DataFrame from data fetching team
            date_column: Name of date/time column
            target_column: Name of target variable column
            cv_initial: Initial training period
            cv_period: Period between cutoff dates
            cv_horizon: Forecast horizon
            **model_kwargs: Additional Prophet model parameters
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Starting pipeline with cross-validation")
        
        # Preprocess data
        processed_data = self.preprocessor.prepare_for_prophet(
            data, date_column, target_column
        )
        
        # Train model
        self.forecaster.create_model(**model_kwargs)
        self.forecaster.fit(processed_data)
        
        # Cross-validate
        logger.info("Running cross-validation...")
        cv_results = self.forecaster.cross_validate(
            processed_data, cv_initial, cv_period, cv_horizon
        )
        
        # Calculate metrics
        metrics = self.forecaster.calculate_metrics(cv_results)
        
        # Visualize CV results
        self.visualizer.plot_cross_validation(cv_results)
        
        logger.info(f"Cross-validation metrics: {metrics}")
        return metrics


# =============================================================================
# EXAMPLE USAGE & ENTRY POINTS
# =============================================================================

def example_basic_forecast():
    """
    Example: Basic forecast using data from another team member.
    """
    # Initialize pipeline
    pipeline = NASAForecastPipeline()
    
    # Load data (assuming data is provided by data fetching team)
    # In real usage, you'd receive this data from another module/file
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2024-11-01', freq='D'),
        'value': np.random.randn(1766).cumsum() + 100  # Example time series
    })
    
    # Run forecast
    historical, forecast = pipeline.run_pipeline(
        data=data,
        date_column='date',
        target_column='value',
        forecast_periods=180,  # 6 months ahead
        frequency='D'
    )
    
    return historical, forecast


def example_with_seasonality():
    """
    Example: Forecast with custom seasonality (e.g., solar cycles).
    """
    pipeline = NASAForecastPipeline()
    
    # Assume data is provided
    # data = load_solar_activity_data()
    
    # Custom model configuration for solar cycle (11-year period)
    historical, forecast = pipeline.run_pipeline(
        data=None,  # Replace with actual data
        date_column='date',
        target_column='sunspot_count',
        forecast_periods=365,
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    
    # Add custom solar cycle seasonality
    # pipeline.forecaster.add_custom_seasonality(
    #     name='solar_cycle',
    #     period=4018,  # 11 years in days
    #     fourier_order=5
    # )
    
    return historical, forecast


def example_with_cross_validation():
    """
    Example: Evaluate forecast accuracy using cross-validation.
    """
    pipeline = NASAForecastPipeline()
    
    # Load data
    data = pd.DataFrame({
        'date': pd.date_range('2015-01-01', '2024-11-01', freq='D'),
        'value': np.random.randn(3592).cumsum() + 100
    })
    
    # Run with cross-validation
    metrics = pipeline.run_with_cross_validation(
        data=data,
        date_column='date',
        target_column='value',
        cv_initial='730 days',   # 2 years initial training
        cv_period='180 days',     # Test every 6 months
        cv_horizon='365 days'     # Forecast 1 year ahead
    )
    
    print(f"Forecast Metrics: {metrics}")
    return metrics


if __name__ == "__main__":
    """
    Main entry point for testing the analysis pipeline.
    """
    logger.info("NASA Data Analysis and Forecasting System")
    logger.info("=" * 60)
    
    print("Pipeline template ready. Implement TODO sections to complete.")
    print("\nRecommended next steps:")
    print("1. Install dependencies: pip install prophet pandas numpy matplotlib seaborn")
    print("2. Receive data format specification from data fetching team")
    print("3. Implement DataPreprocessor methods")
    print("4. Complete ProphetForecaster implementation")
    print("5. Test with sample data from examples above")
    print("6. Coordinate with data fetching team for integration")
    print("\nYour responsibilities:")
    print("- Data preprocessing and cleaning")
    print("- Prophet model configuration and training")
    print("- Forecast generation and evaluation")
    print("- Visualization of results")
