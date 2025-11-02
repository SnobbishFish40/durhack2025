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
from prophet import Prophet
import matplotlib.pyplot as plt
# import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_data(filepath: str) -> pd.DataFrame:
    header_end_line = 0
    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            if line.strip() == '-END HEADER-':
                header_end_line = i + 1  # skip this line as well
                break
    try:
        df = pd.read_csv(filepath, skiprows=header_end_line)
        logger.info("CSV loaded successfully")
        logger.info(f"DataFrame head:\n{df.head()}")
        return df
    except Exception as e:
        logger.error("Error loading CSV:", e)
        return pd.DataFrame()  # Return empty DataFrame on error

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """
    Handles data cleaning, transformation, and preparation for Prophet.
    
    Prophet requires data in specific format:
    - Column 'ds' for dates (datetime)
    - Column 'y' for the target variable (numeric)
    - Optional: additional regressors as separate columns
    
    Note: NASA data often has dates split as year + day-of-year (1-365/366)
    """
    
    def __init__(self, filepath: str):
        self.df = load_data(filepath)
        self.scaler = None  # For normalization if needed
        self.original_columns = None  # Store original column names
    
    def year_doy_to_datetime(self, df: pd.DataFrame,
                            year_column: str,
                            doy_column: str) -> pd.Series:
        """
        Convert year and day-of-year columns to datetime.
        NASA data typically uses this format: year (e.g., 2024) and DOY (1-365/366).
        
        Args:
            df: Input DataFrame
            year_column: Name of the year column
            doy_column: Name of the day-of-year column (1-365 or 1-366 for leap years)
            
        Returns:
            pandas Series with datetime values
            
        Example:
            year=2024, doy=1 -> 2024-01-01
            year=2024, doy=32 -> 2024-02-01
            year=2024, doy=365 -> 2024-12-30 (2024 is leap year, so 366 total)
        """

        logger.info(f"Converting year ({year_column}) and day-of-year ({doy_column}) to datetime")
        
        # Create datetime from year and day of year
        # Format: year + timedelta(days = doy - 1)
        dates = pd.to_datetime(df[year_column].astype(str) + '-01-01') + \
                pd.to_timedelta(df[doy_column] - 1, unit='D')
        
        logger.info(f"Converted date range: {dates.min()} to {dates.max()}")
        return dates
        
    def prepare_for_prophet(self, df: pd.DataFrame, 
                           target_column: str,
                           date_column: Optional[str] = None,
                           year_column: Optional[str] = None,
                           doy_column: Optional[str] = None,
                           handle_missing: bool = True,
                           remove_outliers: bool = False,
                           outlier_method: str = 'iqr',
                           outlier_threshold: float = 1.5) -> pd.DataFrame:
        """
        Transform raw data into Prophet-compatible format.
        Supports both standard date columns and NASA's year+day-of-year format.
        
        Args:
            df: Input DataFrame
            target_column: Name of the target variable column
            date_column: Name of the date/time column (if using single date column)
            year_column: Name of the year column (if using year+DOY format)
            doy_column: Name of the day-of-year column (if using year+DOY format)
            handle_missing: Whether to automatically handle missing values
            remove_outliers: Whether to detect and remove outliers (default False)
            outlier_method: Method for outlier detection - 'iqr' or 'zscore' (default 'iqr')
            outlier_threshold: Threshold for outlier detection (default 1.5 for IQR, 3.0 for zscore)
            
        Returns:
            DataFrame with 'ds' and 'y' columns, sorted and cleaned
            
        Examples:
            # Option 1: Single date column
            prepare_for_prophet(df, target_column='temp', date_column='date')
            
            # Option 2: Year + Day-of-year columns (NASA format)
            prepare_for_prophet(df, target_column='temp', year_column='year', doy_column='doy')
            
            # Option 3: With outlier removal
            prepare_for_prophet(df, target_column='temp', year_column='year', doy_column='doy',
                              remove_outliers=True, outlier_method='iqr', outlier_threshold=1.5)
        """
        logger.info(f"Preparing data for Prophet (input shape: {df.shape})")
        
        # Store original columns for reference
        self.original_columns = df.columns.tolist()
        
        # Create a copy to avoid modifying original
        result = df.copy()
        
        # 1. Handle date conversion based on input format
        if year_column and doy_column:
            # NASA format: year + day-of-year
            if year_column not in result.columns:
                raise ValueError(f"Year column '{year_column}' not found in DataFrame")
            if doy_column not in result.columns:
                raise ValueError(f"Day-of-year column '{doy_column}' not found in DataFrame")
            
            date_series = self.year_doy_to_datetime(result, year_column, doy_column)
            
        elif date_column:
            # Standard format: single date column
            if date_column not in result.columns:
                raise ValueError(f"Date column '{date_column}' not found in DataFrame")
            
            if not pd.api.types.is_datetime64_any_dtype(result[date_column]):
                logger.info(f"Converting {date_column} to datetime")
                date_series = pd.to_datetime(result[date_column], errors='coerce')
            else:
                date_series = result[date_column]
        else:
            raise ValueError("Must provide either 'date_column' OR both 'year_column' and 'doy_column'")
        
        # 2. Check for invalid dates
        invalid_dates = date_series.isna().sum()
        if invalid_dates > 0:
            logger.warning(f"Found {invalid_dates} invalid dates, removing them")
            valid_mask = date_series.notna()
            date_series = date_series[valid_mask]
            result = result[valid_mask]
        
        # 3. Ensure target column is numeric
        if target_column not in result.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        if not pd.api.types.is_numeric_dtype(result[target_column]):
            logger.info(f"Converting {target_column} to numeric")
            result[target_column] = pd.to_numeric(result[target_column], errors='coerce')
        
        # 4. Create Prophet format with 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            'ds': date_series.values,
            'y': result[target_column].values
        })
        
        # 5. Handle missing values in target

        missing_count = (prophet_df['y']==-999).sum()
        if missing_count > 0:
            logger.warning(f"Found {missing_count} missing values in target column")
            if handle_missing:
                prophet_df = self.handle_missing_values(prophet_df, method='interpolate')
            else:
                logger.info("Dropping rows with missing target values")
                prophet_df = prophet_df.dropna(subset=['y'])
        
        # 6. Remove outliers if requested
        if remove_outliers:
            logger.info(f"Detecting outliers using {outlier_method} method (threshold={outlier_threshold})")
            prophet_df = self.detect_outliers(
                df=prophet_df,
                column='y',
                method=outlier_method,
                threshold=outlier_threshold,
                action='remove'  # Remove outliers
            )
        
        # 7. Sort by date
        prophet_df = prophet_df.sort_values('ds').reset_index(drop=True)
        
        # 8. Remove duplicates (keep first occurrence)
        duplicates = prophet_df.duplicated(subset=['ds']).sum()
        if duplicates > 0:
            logger.warning(f"Found {duplicates} duplicate dates, keeping first occurrence")
            prophet_df = prophet_df.drop_duplicates(subset=['ds'], keep='first')
        
        # 9. Check for reasonable data
        if len(prophet_df) < 2:
            raise ValueError(f"Insufficient data after preprocessing: only {len(prophet_df)} rows")
        
        logger.info(f"Data prepared for Prophet (output shape: {prophet_df.shape})")
        logger.info(f"Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
        logger.info(f"Target range: {prophet_df['y'].min():.2f} to {prophet_df['y'].max():.2f}")
        
        return prophet_df
    
    def handle_missing_values(self, df: pd.DataFrame, 
                             method: str = 'drop',
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
        missing_before = (result[column]==-999).sum()
        
        if missing_before == 0:
            return result
        
        logger.info(f"Handling {missing_before} missing values using method: {method}")
        
        if method == 'interpolate':
            # Replace -999 with NaN, then interpolate
            result[column] = result[column].replace(-999, np.nan)
            result[column] = result[column].interpolate(method='linear', limit_direction='both')

        elif method == 'forward_fill':
            result[column] = result[column].replace(-999, np.nan)
            result[column] = result[column].fillna(method='ffill')

        elif method == 'backward_fill':
            result[column] = result[column].replace(-999, np.nan)
            result[column] = result[column].fillna(method='bfill')

        elif method == 'mean':
            result[column] = result[column].replace(-999, np.nan)
            result[column] = result[column].fillna(result[column].mean())

        elif method == 'drop':
            # Drop rows where the value is NaN or -999
            result = result.dropna(subset=[column])
            result = result[result[column] != -999]

        else:
            raise ValueError(f"Unknown method: {method}")
        
        missing_after = (result[column]==-999).sum()
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
            print(Q1, Q3)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask = (result[column] < lower_bound) | (result[column] > upper_bound)
            print(result)
            print(outlier_mask)
            
        elif method == 'zscore':
            # Z-score method
            mean = result[column].mean()
            std = result[column].std()
            z_scores = np.abs((result[column] - mean) / std)
            outlier_mask = z_scores > threshold
            print(result)
            print(outlier_mask)
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
# PROPHET MODEL CONFIGURATION & TRAINING
# =============================================================================

class ProphetForecaster:
    """
    Wrapper for Facebook Prophet with NASA-specific configurations.
    Handles model creation, training, prediction, and evaluation.
    """
    
    def __init__(self, seasonality_mode: str = 'multiplicative',
                 changepoint_prior_scale: float = 0.05):
        """
        Initialize Prophet model with configuration.
        
        Args:
            seasonality_mode: 'additive' or 'multiplicative'
                - 'additive': seasonality + trend (use when seasonal variations are constant)
                - 'multiplicative': seasonality * trend (use when variations grow with trend)
            changepoint_prior_scale: Flexibility of trend changes (0.001-0.5)
                - Lower values = less flexible, smoother trend
                - Higher values = more flexible, can capture rapid changes
                - Default 0.05 works well for most cases
        """
        self.model = None
        self.seasonality_mode = seasonality_mode
        self.changepoint_prior_scale = changepoint_prior_scale
        self.fitted = False
        self.training_data = None
        
    def create_model(self, **kwargs) -> None:
        """
        Create and configure Prophet model.
        
        Common configurations for NASA data:
        - yearly_seasonality: For annual patterns (e.g., orbital cycles, seasons)
        - weekly_seasonality: For weekly patterns
        - daily_seasonality: For daily patterns (use for hourly data)
        - holidays: Special events (solar eclipses, meteor showers, etc.)
        - interval_width: Confidence interval width (default 0.80 = 80%)
        
        Additional useful parameters:
        - growth: 'linear' (default) or 'logistic' (for saturating growth)
        - n_changepoints: Number of potential changepoints (default 25)
        - changepoint_range: Proportion of history for changepoints (default 0.8)
        
        Args:
            **kwargs: Additional Prophet parameters to override defaults
        """
        try:
            from prophet import Prophet
        except ImportError:
            raise ImportError(
                "Prophet not installed. Run: pip install prophet\n"
                "Note: Prophet requires pystan. If issues occur, try:\n"
                "  pip install pystan==2.19.1.1\n"
                "  pip install prophet"
            )
        
        # Default settings optimized for NASA time series data
        default_params = {
            'seasonality_mode': self.seasonality_mode,
            'changepoint_prior_scale': self.changepoint_prior_scale,
            'yearly_seasonality': True,   # Most NASA data has yearly patterns
            'weekly_seasonality': False,  # Usually not relevant for NASA data
            'daily_seasonality': False,   # Usually not relevant unless hourly data
            'interval_width': 0.80,       # 80% confidence intervals
        }
        
        # Override defaults with user-provided parameters
        default_params.update(kwargs)
        
        logger.info(f"Creating Prophet model with parameters:")
        for key, value in default_params.items():
            logger.info(f"  {key}: {value}")
        
        self.model = Prophet(**default_params)
        logger.info("Prophet model created successfully")
    
    def add_custom_seasonality(self, name: str, period: float, 
                              fourier_order: int, prior_scale: float = 10.0) -> None:
        """
        Add custom seasonality patterns specific to NASA data.
        
        Examples for NASA data:
        - Lunar cycle: period=29.53 days, fourier_order=5
        - Solar cycle: period=4018 days (11 years), fourier_order=10
        - Mars year: period=687 days, fourier_order=8
        - El Niño cycle: period=1460 days (4 years), fourier_order=6
        
        Args:
            name: Name of the seasonality (e.g., 'lunar_cycle', 'solar_cycle')
            period: Period in days
            fourier_order: Number of Fourier terms (higher = more complex pattern)
                - Low (3-5): simple, smooth patterns
                - Medium (6-10): moderate complexity
                - High (11+): complex, irregular patterns
            prior_scale: Strength of seasonality (default 10.0)
                - Higher values = stronger seasonal effects
        """
        if self.model is None:
            raise ValueError("Must call create_model() before adding seasonality")
        
        self.model.add_seasonality(
            name=name,
            period=period,
            fourier_order=fourier_order,
            prior_scale=prior_scale
        )
        
        logger.info(f"Added custom seasonality: {name} (period={period} days, fourier_order={fourier_order})")
    
    def add_regressor(self, name: str, prior_scale: float = 10.0, 
                     standardize: bool = True) -> None:
        """
        Add external regressor variable to the model.
        Must be called BEFORE fitting, and regressor data must be in the dataframe.
        
        Examples for NASA data:
        - Solar flux index
        - Atmospheric pressure
        - Sea surface temperature
        - Vegetation index
        
        Args:
            name: Name of the regressor column in the dataframe
            prior_scale: Strength of the regressor effect (default 10.0)
            standardize: Whether to standardize the regressor (default True)
        """
        if self.model is None:
            raise ValueError("Must call create_model() before adding regressors")
        
        self.model.add_regressor(
            name=name,
            prior_scale=prior_scale,
            standardize=standardize
        )
        
        logger.info(f"Added regressor: {name} (prior_scale={prior_scale}, standardize={standardize})")
    
    def add_holidays(self, holidays_df: pd.DataFrame) -> None:
        """
        Add special events/holidays to the model.
        
        For NASA data, this could include:
        - Solar eclipses
        - Meteor showers
        - Satellite launch dates
        - Major solar flares
        
        Args:
            holidays_df: DataFrame with required columns:
                - 'holiday': name of the event
                - 'ds': date of the event
                Optional columns:
                - 'lower_window': days before event to include (default 0)
                - 'upper_window': days after event to include (default 0)
                - 'prior_scale': strength of holiday effect (default 10.0)
        
        Example:
            holidays = pd.DataFrame({
                'holiday': ['Solar Eclipse', 'Solar Eclipse'],
                'ds': pd.to_datetime(['2024-04-08', '2024-10-02']),
                'lower_window': 0,
                'upper_window': 1
            })
        """
        if self.model is None:
            raise ValueError("Must call create_model() before adding holidays")
        
        required_cols = ['holiday', 'ds']
        if not all(col in holidays_df.columns for col in required_cols):
            raise ValueError(f"holidays_df must contain columns: {required_cols}")
        
        self.model = Prophet(holidays=holidays_df)
        logger.info(f"Added {len(holidays_df)} holiday events")
    
    def fit(self, df: pd.DataFrame, **fit_kwargs) -> None:
        """
        Train the Prophet model on historical data.
        
        Args:
            df: Prophet-formatted DataFrame with 'ds' and 'y' columns
                - 'ds': datetime column
                - 'y': target variable (numeric)
                - Any additional regressor columns (if added with add_regressor)
            **fit_kwargs: Additional fitting parameters (rarely needed)
        
        Raises:
            ValueError: If model not created or data format invalid
        """
        if self.model is None:
            logger.warning("Model not created, creating with default parameters")
            self.create_model()
        
        # Validate input data
        if 'ds' not in df.columns or 'y' not in df.columns:
            raise ValueError("DataFrame must have 'ds' and 'y' columns")
        
        if len(df) < 2:
            raise ValueError(f"Need at least 2 data points to fit, got {len(df)}")
        
        logger.info(f"Training Prophet model on {len(df)} data points")
        logger.info(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
        
        # Store training data for later use
        self.training_data = df.copy()
        
        # Fit the model
        try:
            self.model.fit(df, **fit_kwargs)
            self.fitted = True
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error(f"Error during model fitting: {e}")
            raise
    
    def predict(self, periods: int, frequency: str = 'D', 
                include_history: bool = True) -> pd.DataFrame:
        """
        Generate future predictions.
        
        Args:
            periods: Number of periods to forecast into the future
            frequency: Forecast frequency
                - 'D': daily (default)
                - 'W': weekly
                - 'M': monthly (month end)
                - 'MS': monthly (month start)
                - 'H': hourly
            include_history: Whether to include historical fitted values (default True)
            
        Returns:
            DataFrame with columns:
                - 'ds': dates
                - 'yhat': predicted values
                - 'yhat_lower': lower bound of confidence interval
                - 'yhat_upper': upper bound of confidence interval
                - 'trend': trend component
                - 'seasonal' components (yearly, weekly, etc.)
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions. Call fit() first.")
        
        logger.info(f"Generating forecast for {periods} {frequency} periods")
        
        # Create future dataframe
        if include_history:
            # Include historical dates + future dates
            future = self.model.make_future_dataframe(periods=periods, freq=frequency)
        else:
            # Only future dates
            last_date = self.training_data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(1, unit='D'),
                periods=periods,
                freq=frequency
            )
            future = pd.DataFrame({'ds': future_dates})
        
        # Add regressor columns if they were used in training
        if self.training_data is not None:
            regressor_cols = [col for col in self.training_data.columns 
                            if col not in ['ds', 'y']]
            if regressor_cols:
                logger.warning(
                    f"Model has regressors {regressor_cols} but future values not provided. "
                    "Forecast will use last known values. For better results, provide future regressor values."
                )
                # Use last known values for regressors (naive approach)
                for col in regressor_cols:
                    last_value = self.training_data[col].iloc[-1]
                    future[col] = last_value
        
        # Generate predictions
        forecast = self.model.predict(future)
        
        logger.info(f"Forecast generated: {len(forecast)} rows")
        logger.info(f"Forecast date range: {forecast['ds'].min()} to {forecast['ds'].max()}")
        
        return forecast
    
    def cross_validate(self, df: pd.DataFrame, 
                      initial: str, period: str, horizon: str,
                      parallel: Optional[str] = None) -> pd.DataFrame:
        """
        Perform time series cross-validation to evaluate model performance.
        
        This simulates forecasting in the past by:
        1. Training on initial period
        2. Forecasting horizon into the future
        3. Moving forward by period and repeating
        
        Args:
            df: Prophet-formatted DataFrame (ds, y columns)
            initial: Initial training period
                Examples: '730 days', '2 years', '365 days'
            period: Period between cutoff dates
                Examples: '180 days', '90 days', '30 days'
            horizon: Forecast horizon (how far to predict)
                Examples: '365 days', '180 days', '30 days'
            parallel: Parallelization method
                - None: sequential (default, safest)
                - 'processes': parallel with multiprocessing
                - 'threads': parallel with threading
        
        Returns:
            DataFrame with columns:
                - 'ds': date
                - 'yhat': predicted value
                - 'yhat_lower': lower bound
                - 'yhat_upper': upper bound
                - 'y': actual value
                - 'cutoff': training data cutoff date
        
        Example:
            # Train on 2 years, forecast 1 year, test every 6 months
            cv_results = forecaster.cross_validate(
                df=data,
                initial='730 days',
                period='180 days',
                horizon='365 days'
            )
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before cross-validation")
        
        try:
            from prophet.diagnostics import cross_validation as prophet_cv
        except ImportError:
            raise ImportError("Prophet diagnostics not available. Update prophet package.")
        
        logger.info(f"Running cross-validation:")
        logger.info(f"  Initial training period: {initial}")
        logger.info(f"  Period between cutoffs: {period}")
        logger.info(f"  Forecast horizon: {horizon}")
        
        cv_results = prophet_cv(
            self.model,
            initial=initial,
            period=period,
            horizon=horizon,
            parallel=parallel
        )
        
        logger.info(f"Cross-validation complete: {len(cv_results)} predictions evaluated")
        
        return cv_results
    
    def calculate_metrics(self, cv_results: pd.DataFrame, 
                         metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Calculate forecast accuracy metrics from cross-validation results.
        
        Args:
            cv_results: DataFrame from cross_validate() method
            metrics: List of metrics to calculate. If None, calculates all:
                - 'mse': Mean Squared Error
                - 'rmse': Root Mean Squared Error
                - 'mae': Mean Absolute Error
                - 'mape': Mean Absolute Percentage Error
                - 'mdape': Median Absolute Percentage Error
                - 'smape': Symmetric Mean Absolute Percentage Error
                - 'coverage': Percentage of actuals within confidence interval
        
        Returns:
            Dictionary of metric names and values
        
        Interpretation:
            - Lower MAE/RMSE/MAPE = better predictions
            - Higher coverage (closer to interval_width) = better uncertainty estimates
        """
        try:
            from prophet.diagnostics import performance_metrics
        except ImportError:
            raise ImportError("Prophet diagnostics not available. Update prophet package.")
        
        logger.info("Calculating performance metrics...")
        
        # Calculate all metrics
        df_metrics = performance_metrics(cv_results, metrics=metrics)
        
        # Get average metrics across all horizons
        metrics_dict = {
            'MAE': df_metrics['mae'].mean(),
            'RMSE': df_metrics['rmse'].mean(),
            'MAPE': df_metrics['mape'].mean(),
            'MDAPE': df_metrics['mdape'].mean(),
            'SMAPE': df_metrics['smape'].mean(),
            'Coverage': df_metrics['coverage'].mean()
        }
        
        logger.info("Performance Metrics:")
        for metric, value in metrics_dict.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics_dict
    
    def get_component_importance(self, forecast: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate the relative importance of different forecast components.
        
        Args:
            forecast: DataFrame from predict() method
            
        Returns:
            Dictionary with component importance percentages
        """
        components = {}
        
        # Get variance of each component
        if 'trend' in forecast.columns:
            components['trend'] = forecast['trend'].var()
        
        if 'yearly' in forecast.columns:
            components['yearly'] = forecast['yearly'].var()
        
        if 'weekly' in forecast.columns:
            components['weekly'] = forecast['weekly'].var()
        
        # Calculate percentages
        total_var = sum(components.values())
        if total_var > 0:
            importance = {k: (v / total_var) * 100 for k, v in components.items()}
        else:
            importance = {k: 0 for k in components.keys()}
        
        logger.info("Component Importance:")
        for comp, pct in importance.items():
            logger.info(f"  {comp}: {pct:.1f}%")
        
        return importance


# =============================================================================
# VISUALIZATION
# =============================================================================

class ForecastVisualizer:
    """
    Create visualizations for Prophet forecasts and NASA data analysis.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-darkgrid', figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer with plotting style.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            self.plt = plt
            try:
                self.plt.style.use(style)
            except:
                logger.warning(f"Style '{style}' not available, using default")
        except ImportError:
            logger.warning("matplotlib not installed - visualization will be disabled")
            self.plt = None
        
        self.figsize = figsize
        logger.info("ForecastVisualizer initialized")
    
    def plot_forecast(self, model, forecast: pd.DataFrame, 
                     historical_data: pd.DataFrame = None,
                     title: str = "NASA Data Forecast",
                     xlabel: str = "Date",
                     ylabel: str = "Value",
                     save_path: Optional[str] = None) -> None:
        """
        Plot forecast with confidence intervals and optional historical data.
        
        Args:
            model: Fitted Prophet model
            forecast: Forecast DataFrame from Prophet
            historical_data: Optional historical data (must have 'ds' and 'y' columns)
            title: Plot title
            xlabel: X-axis label
            ylabel: Y-axis label
            save_path: Optional path to save figure
        """
        if self.plt is None:
            logger.error("matplotlib not available")
            return
        
        fig = model.plot(forecast, figsize=self.figsize)
        ax = fig.gca()
        
        # Overlay historical data if provided
        if historical_data is not None:
            ax.plot(historical_data['ds'], historical_data['y'], 
                   'k.', markersize=3, alpha=0.5, label='Historical Data')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast plot saved to {save_path}")
        
        self.plt.show()
    
    def plot_components(self, model, forecast: pd.DataFrame,
                       figsize: Optional[Tuple[int, int]] = None,
                       save_path: Optional[str] = None) -> None:
        """
        Plot forecast components (trend, seasonality, holidays).
        
        Args:
            model: Fitted Prophet model
            forecast: Forecast DataFrame
            figsize: Optional custom figure size
            save_path: Optional path to save figure
        """
        if self.plt is None:
            logger.error("matplotlib not available")
            return
        
        fig_size = figsize if figsize else (12, 8)
        fig = model.plot_components(forecast, figsize=fig_size)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Components plot saved to {save_path}")
        
        self.plt.show()
    
    def plot_cross_validation(self, cv_results: pd.DataFrame,
                             metric: str = 'mape',
                             rolling_window: float = 0.1,
                             title: Optional[str] = None,
                             save_path: Optional[str] = None) -> None:
        """
        Visualize cross-validation results with performance metrics over time.
        
        Args:
            cv_results: Cross-validation results from Prophet
            metric: Metric to plot ('mape', 'rmse', 'mae', 'coverage')
            rolling_window: Proportion of data for rolling window (0.0 to 1.0)
            title: Optional custom title
            save_path: Optional path to save figure
        """
        if self.plt is None:
            logger.error("matplotlib not available")
            return
        
        try:
            from prophet.diagnostics import performance_metrics
            from prophet.plot import plot_cross_validation_metric
            
            # Calculate performance metrics
            df_p = performance_metrics(cv_results, rolling_window=rolling_window)
            
            # Create plot
            fig = plot_cross_validation_metric(cv_results, metric=metric, 
                                              rolling_window=rolling_window,
                                              figsize=self.figsize)
            
            if title:
                self.plt.title(title, fontsize=14, fontweight='bold')
            else:
                self.plt.title(f'Cross-Validation: {metric.upper()}', 
                             fontsize=14, fontweight='bold')
            
            self.plt.grid(True, alpha=0.3)
            self.plt.tight_layout()
            
            if save_path:
                self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Cross-validation plot saved to {save_path}")
            
            self.plt.show()
            
        except ImportError:
            logger.error("Prophet diagnostics not available")
    
    def plot_prediction_intervals(self, forecast: pd.DataFrame,
                                 historical_data: Optional[pd.DataFrame] = None,
                                 title: str = "Prediction Intervals",
                                 save_path: Optional[str] = None) -> None:
        """
        Plot prediction intervals showing uncertainty over time.
        
        Args:
            forecast: Forecast DataFrame with yhat, yhat_lower, yhat_upper
            historical_data: Optional historical data
            title: Plot title
            save_path: Optional path to save figure
        """
        if self.plt is None:
            logger.error("matplotlib not available")
            return
        
        fig, ax = self.plt.subplots(figsize=self.figsize)
        
        # Plot historical data if available
        if historical_data is not None:
            ax.plot(historical_data['ds'], historical_data['y'],
                   'k.', markersize=2, alpha=0.4, label='Historical')
        
        # Plot forecast
        ax.plot(forecast['ds'], forecast['yhat'], 
               'b-', linewidth=2, label='Forecast')
        
        # Plot confidence intervals
        ax.fill_between(forecast['ds'], 
                       forecast['yhat_lower'],
                       forecast['yhat_upper'],
                       alpha=0.2, color='blue', label='80% Interval')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Prediction intervals plot saved to {save_path}")
        
        self.plt.show()
    
    def plot_residuals(self, historical_data: pd.DataFrame,
                      forecast: pd.DataFrame,
                      title: str = "Residual Analysis",
                      save_path: Optional[str] = None) -> None:
        """
        Plot residuals to assess model fit quality.
        
        Args:
            historical_data: Historical data with 'ds' and 'y' columns
            forecast: Forecast DataFrame
            title: Plot title
            save_path: Optional path to save figure
        """
        if self.plt is None:
            logger.error("matplotlib not available")
            return
        
        # Merge data to calculate residuals
        merged = historical_data.merge(forecast[['ds', 'yhat']], on='ds', how='inner')
        merged['residual'] = merged['y'] - merged['yhat']
        
        fig, axes = self.plt.subplots(2, 2, figsize=(14, 10))
        
        # Residuals over time
        axes[0, 0].scatter(merged['ds'], merged['residual'], alpha=0.5)
        axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Residuals Over Time')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Residual')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals vs predicted
        axes[0, 1].scatter(merged['yhat'], merged['residual'], alpha=0.5)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Residuals vs Predicted')
        axes[0, 1].set_xlabel('Predicted Value')
        axes[0, 1].set_ylabel('Residual')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Histogram of residuals
        axes[1, 0].hist(merged['residual'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Residual Distribution')
        axes[1, 0].set_xlabel('Residual')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(merged['residual'], dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot')
        axes[1, 1].grid(True, alpha=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
        self.plt.tight_layout()
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Residuals plot saved to {save_path}")
        
        self.plt.show()
    
    def create_dashboard(self, model, forecast: pd.DataFrame,
                        historical_data: pd.DataFrame,
                        cv_results: Optional[pd.DataFrame] = None,
                        title: str = "NASA Forecast Dashboard",
                        save_path: Optional[str] = None) -> None:
        """
        Create comprehensive dashboard with multiple plots.
        
        Args:
            model: Fitted Prophet model
            forecast: Forecast DataFrame
            historical_data: Historical data
            cv_results: Optional cross-validation results
            title: Dashboard title
            save_path: Optional path to save figure
        """
        if self.plt is None:
            logger.error("matplotlib not available")
            return
        
        # Determine layout based on CV results
        if cv_results is not None:
            fig = self.plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        else:
            fig = self.plt.figure(figsize=(18, 8))
            gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Main forecast plot
        ax1 = fig.add_subplot(gs[0, :])
        model.plot(forecast, ax=ax1)
        ax1.plot(historical_data['ds'], historical_data['y'],
                'k.', markersize=2, alpha=0.4, label='Historical')
        ax1.set_title('Forecast with Confidence Intervals', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trend component
        ax2 = fig.add_subplot(gs[1, 0])
        if 'trend' in forecast.columns:
            ax2.plot(forecast['ds'], forecast['trend'], 'g-', linewidth=2)
            ax2.set_title('Trend Component', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Trend')
            ax2.grid(True, alpha=0.3)
        
        # 3. Seasonality component
        ax3 = fig.add_subplot(gs[1, 1])
        if 'yearly' in forecast.columns:
            ax3.plot(forecast['ds'], forecast['yearly'], 'orange', linewidth=2)
            ax3.set_title('Yearly Seasonality', fontsize=12, fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Yearly')
            ax3.grid(True, alpha=0.3)
        
        # 4. Cross-validation (if available)
        if cv_results is not None:
            try:
                from prophet.diagnostics import performance_metrics
                df_p = performance_metrics(cv_results, rolling_window=0.1)
                
                ax4 = fig.add_subplot(gs[2, 0])
                ax4.plot(df_p['horizon'], df_p['mape'], 'r-', linewidth=2)
                ax4.set_title('MAPE vs Forecast Horizon', fontsize=12, fontweight='bold')
                ax4.set_xlabel('Horizon')
                ax4.set_ylabel('MAPE')
                ax4.grid(True, alpha=0.3)
                
                ax5 = fig.add_subplot(gs[2, 1])
                ax5.plot(df_p['horizon'], df_p['rmse'], 'b-', linewidth=2)
                ax5.set_title('RMSE vs Forecast Horizon', fontsize=12, fontweight='bold')
                ax5.set_xlabel('Horizon')
                ax5.set_ylabel('RMSE')
                ax5.grid(True, alpha=0.3)
            except:
                logger.warning("Could not plot cross-validation metrics")
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.99)
        
        if save_path:
            self.plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Dashboard saved to {save_path}")
        
        self.plt.show()


# =============================================================================
# MAIN PIPELINE
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
                    target_column: str,
                    date_column: Optional[str] = None,
                    year_column: Optional[str] = None,
                    doy_column: Optional[str] = None,
                    region_filter: Optional['RegionFilter'] = None,
                    forecast_periods: int = 365,
                    frequency: str = 'D',
                    **model_kwargs) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run complete analysis pipeline on provided data with optional region filtering.
        Supports both standard date format and NASA's year+day-of-year format.
        
        Args:
            data: Input DataFrame from data fetching team
            target_column: Name of target variable column
            date_column: Name of date/time column (if using single date column)
            year_column: Name of year column (if using year+DOY format)
            doy_column: Name of day-of-year column (if using year+DOY format)
            region_filter: Optional RegionFilter to limit analysis to specific area/time
            forecast_periods: Number of periods to forecast
            frequency: Forecast frequency ('D', 'W', 'M', 'H')
            **model_kwargs: Additional Prophet model parameters
            
        Returns:
            Tuple of (processed_historical_data, forecast_data)
            
        Examples:
            # NASA format with year + day-of-year
            historical, forecast = pipeline.run_pipeline(
                data=nasa_data,
                target_column='temperature',
                year_column='year',
                doy_column='doy',
                forecast_periods=180
            )
            
            # Standard format with single date column
            historical, forecast = pipeline.run_pipeline(
                data=nasa_data,
                target_column='temperature',
                date_column='date',
                forecast_periods=180
            )
        """
        logger.info("Starting NASA forecast pipeline")
        self.current_region = region_filter
        
        # Step 0: Apply region filter if provided
        working_data = data
        if region_filter:
            logger.info(f"Applying region filter: {region_filter}")
            # For year+doy format, we need to create a temporary date column for filtering
            if year_column and doy_column:
                temp_dates = self.preprocessor.year_doy_to_datetime(data, year_column, doy_column)
                temp_df = data.copy()
                temp_df['_temp_date'] = temp_dates
                working_data = region_filter.apply_all_filters(
                    temp_df,
                    date_col='_temp_date',
                    lat_col='latitude' if 'latitude' in data.columns else None,
                    lon_col='longitude' if 'longitude' in data.columns else None
                )
                working_data = working_data.drop(columns=['_temp_date'])
            else:
                working_data = region_filter.apply_all_filters(
                    data, 
                    date_col=date_column,
                    lat_col='latitude' if 'latitude' in data.columns else None,
                    lon_col='longitude' if 'longitude' in data.columns else None
                )
            
            if len(working_data) == 0:
                raise ValueError("Region filter resulted in empty dataset. Adjust filter parameters.")
        
        # Determine required columns for validation
        required_cols = [target_column]
        if date_column:
            required_cols.append(date_column)
        elif year_column and doy_column:
            required_cols.extend([year_column, doy_column])
        
        # Step 1: Validate input data
        logger.info("Validating input data...")
        self.loader.validate_data(working_data, required_columns=required_cols)
        
        # Step 2: Preprocess
        logger.info("Preprocessing data...")
        processed_data = self.preprocessor.prepare_for_prophet(
            working_data,
            target_column=target_column,
            date_column=date_column,
            year_column=year_column,
            doy_column=doy_column
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
        title = f"Forecast for {region_filter.location_name}" if region_filter else "NASA Data Forecast"
        self.visualizer.plot_forecast(
            self.forecaster.model, 
            forecast, 
            processed_data,
            title=title
        )
        self.visualizer.plot_components(self.forecaster.model, forecast)
        
        logger.info("Pipeline complete!")
        return processed_data, forecast
    
    def run_with_cross_validation(self, data: pd.DataFrame,
                                  target_column: str,
                                  date_column: Optional[str] = None,
                                  year_column: Optional[str] = None,
                                  doy_column: Optional[str] = None,
                                  region_filter: Optional['RegionFilter'] = None,
                                  cv_initial: str = '730 days',
                                  cv_period: str = '180 days',
                                  cv_horizon: str = '365 days',
                                  **model_kwargs) -> Dict[str, float]:
        """
        Run pipeline with cross-validation to evaluate forecast accuracy.
        Supports both standard date format and NASA's year+day-of-year format.
        
        Args:
            data: Input DataFrame from data fetching team
            target_column: Name of target variable column
            date_column: Name of date/time column (if using single date column)
            year_column: Name of year column (if using year+DOY format)
            doy_column: Name of day-of-year column (if using year+DOY format)
            region_filter: Optional RegionFilter to limit analysis
            cv_initial: Initial training period
            cv_period: Period between cutoff dates
            cv_horizon: Forecast horizon
            **model_kwargs: Additional Prophet model parameters
            
        Returns:
            Dictionary of performance metrics
        """
        logger.info("Starting pipeline with cross-validation")
        self.current_region = region_filter
        
        # Apply region filter if provided
        working_data = data
        if region_filter:
            logger.info(f"Applying region filter: {region_filter}")
            # Handle year+doy format for filtering
            if year_column and doy_column:
                temp_dates = self.preprocessor.year_doy_to_datetime(data, year_column, doy_column)
                temp_df = data.copy()
                temp_df['_temp_date'] = temp_dates
                working_data = region_filter.apply_all_filters(
                    temp_df,
                    date_col='_temp_date',
                    lat_col='latitude' if 'latitude' in data.columns else None,
                    lon_col='longitude' if 'longitude' in data.columns else None
                )
                working_data = working_data.drop(columns=['_temp_date'])
            else:
                working_data = region_filter.apply_all_filters(
                    data,
                    date_col=date_column,
                    lat_col='latitude' if 'latitude' in data.columns else None,
                    lon_col='longitude' if 'longitude' in data.columns else None
                )
        
        # Preprocess data
        processed_data = self.preprocessor.prepare_for_prophet(
            working_data,
            target_column=target_column,
            date_column=date_column,
            year_column=year_column,
            doy_column=doy_column
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

def example_nasa_year_doy_format():
    """
    Example: Forecast using NASA's year + day-of-year format.
    This is the typical format for NASA Earth observation data.
    
    NASA data typically has:
    - 'year' column: 2020, 2021, 2022, etc.
    - 'doy' column: 1-365 (or 1-366 for leap years)
    
    This example shows how to use the preprocessor directly.
    """
    # Create sample NASA-format data
    years = []
    doys = []
    for year in range(2020, 2025):
        days_in_year = 366 if year % 4 == 0 else 365
        years.extend([year] * days_in_year)
        doys.extend(range(1, days_in_year + 1))
    
    nasa_data = pd.DataFrame({
        'year': years,
        'doy': doys,  # Day of year: 1-365 (or 1-366 for leap years)
        'latitude': np.random.uniform(32, 42, len(years)),
        'longitude': np.random.uniform(-125, -114, len(years)),
        'temperature': np.random.randn(len(years)).cumsum() + 20
    })
    
    print("Sample NASA data format:")
    print(nasa_data.head(10))
    print(f"\nData shape: {nasa_data.shape}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Convert to Prophet format using year + doy columns
    prophet_data = preprocessor.prepare_for_prophet(
        df=nasa_data,
        target_column='temperature',
        year_column='year',  # Specify year column
        doy_column='doy'      # Specify day-of-year column
    )
    
    print("\nProphet-formatted data:")
    print(prophet_data.head(10))
    print(f"\nDate range: {prophet_data['ds'].min()} to {prophet_data['ds'].max()}")
    
    return prophet_data


def example_basic_forecast():
    """
    Example: Basic forecast using standard date column format.
    """
    # Create sample data with standard date column
    data = pd.DataFrame({
        'date': pd.date_range('2020-01-01', '2024-11-01', freq='D'),
        'value': np.random.randn(1766).cumsum() + 100
    })
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor()
    
    # Convert to Prophet format using date column
    prophet_data = preprocessor.prepare_for_prophet(
        df=data,
        target_column='value',
        date_column='date'
    )
    
    print("Standard date format processed:")
    print(prophet_data.head())
    
    return prophet_data


def example_complete_workflow():
    """
    Example: Complete workflow from NASA data to forecast.
    This demonstrates the full ProphetForecaster usage.
    """
    print("\n" + "="*60)
    print("COMPLETE WORKFLOW EXAMPLE")
    print("="*60)
    
    # Step 1: Create sample NASA data
    print("\n1. Creating sample NASA data (year + DOY format)...")
    years = []
    doys = []
    for year in range(2020, 2025):
        days_in_year = 366 if year % 4 == 0 else 365
        years.extend([year] * days_in_year)
        doys.extend(range(1, days_in_year + 1))
    
    # Simulate temperature with trend and seasonality
    n = len(years)
    trend = np.linspace(20, 22, n)
    seasonal = 5 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.randn(n) * 0.5
    temperature = trend + seasonal + noise
    
    nasa_data = pd.DataFrame({
        'year': years,
        'doy': doys,
        'temperature': temperature
    })
    
    # Step 2: Preprocess data
    print("2. Preprocessing data...")
    preprocessor = DataPreprocessor()
    prophet_data = preprocessor.prepare_for_prophet(
        df=nasa_data,
        target_column='temperature',
        year_column='year',
        doy_column='doy'
    )
    
    # Step 3: Create and configure Prophet model
    print("\n3. Creating Prophet model...")
    forecaster = ProphetForecaster(
        seasonality_mode='additive',
        changepoint_prior_scale=0.05
    )
    
    forecaster.create_model(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False
    )
    
    # Step 4: Train the model
    print("4. Training model...")
    forecaster.fit(prophet_data)
    
    # Step 5: Make predictions
    print("\n5. Generating 180-day forecast...")
    forecast = forecaster.predict(periods=180, frequency='D')
    
    print(f"\nForecast shape: {forecast.shape}")
    print("\nForecast sample (last 5 future days):")
    print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
    
    # Step 6: Evaluate with cross-validation
    print("\n6. Running cross-validation...")
    try:
        cv_results = forecaster.cross_validate(
            df=prophet_data,
            initial='365 days',
            period='90 days',
            horizon='90 days'
        )
        
        # Calculate metrics
        metrics = forecaster.calculate_metrics(cv_results)
        print("\nForecast Accuracy Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    except Exception as e:
        print(f"Cross-validation skipped: {e}")
    
    # Step 7: Component importance
    importance = forecaster.get_component_importance(forecast)
    
    return prophet_data, forecast


def example_with_custom_seasonality():
    """
    Example: Add custom seasonality for NASA-specific cycles.
    """
    print("\n" + "="*60)
    print("CUSTOM SEASONALITY EXAMPLE")
    print("="*60)
    
    # Create sample data
    dates = pd.date_range('2015-01-01', '2024-11-01', freq='D')
    n = len(dates)
    
    # Simulate data with 11-year solar cycle
    solar_cycle = 2 * np.sin(2 * np.pi * np.arange(n) / 4018)  # 4018 days = ~11 years
    yearly = 3 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    trend = np.linspace(50, 55, n)
    noise = np.random.randn(n) * 0.5
    
    data = pd.DataFrame({
        'ds': dates,
        'y': trend + solar_cycle + yearly + noise
    })
    
    print(f"Data: {len(data)} days from {data['ds'].min()} to {data['ds'].max()}")
    
    # Create forecaster with custom solar cycle
    forecaster = ProphetForecaster(seasonality_mode='additive')
    forecaster.create_model(
        yearly_seasonality=True,
        weekly_seasonality=False
    )
    
    # Add 11-year solar cycle
    print("\nAdding solar cycle seasonality (11-year period)...")
    forecaster.add_custom_seasonality(
        name='solar_cycle',
        period=4018,  # 11 years in days
        fourier_order=8
    )
    
    # Add lunar cycle
    print("Adding lunar cycle seasonality (29.53-day period)...")
    forecaster.add_custom_seasonality(
        name='lunar_cycle',
        period=29.53,
        fourier_order=5
    )
    
    # Fit and predict
    print("\nTraining model...")
    forecaster.fit(data)
    
    print("Generating 3-year forecast...")
    forecast = forecaster.predict(periods=365*3, frequency='D')
    
    print(f"\nForecast generated: {len(forecast)} days")
    print("\nSample forecast:")
    print(forecast[['ds', 'yhat', 'trend', 'yearly', 'solar_cycle', 'lunar_cycle']].tail(10))
    
    return data, forecast
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
    
    print("\n" + "="*60)
    print("NASA DATA ANALYSIS & FORECASTING SYSTEM")
    print("="*60)
    print("\n✅ FULLY IMPLEMENTED CLASSES:")
    print("  1. DataPreprocessor - Complete with all methods")
    print("  2. ProphetForecaster - Complete with all methods")
    print("\n📊 DATA FORMAT SUPPORT:")
    print("  ✓ NASA year+DOY format: year_column='year', doy_column='doy'")
    print("  ✓ Standard date format: date_column='date'")
    print("\n🚀 READY TO USE!")
    print("\nQuick start:")
    print("  1. Install: pip install prophet pandas numpy matplotlib")
    print("  2. Get NASA data from your teammate")
    print("  3. Run example_complete_workflow() to see full demo")
    print("\n" + "="*60)
    print("AVAILABLE EXAMPLES:")
    print("="*60)
    print("  example_nasa_year_doy_format()  - NASA date format demo")
    print("  example_complete_workflow()     - Full end-to-end workflow")
    print("  example_with_custom_seasonality() - Solar/lunar cycles")
    print("\nUncomment below to run examples:")
    print("="*60)
    
    # Uncomment to test:
    # example_complete_workflow()
    # example_with_custom_seasonality()
    print("\nYour responsibilities:")
    print("- Data preprocessing and cleaning")
    print("- Prophet model configuration and training")
    print("- Forecast generation and evaluation")
    print("- Visualization of results")
