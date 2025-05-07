import numpy as np
import pandas as pd


def calculate_nutritional_status(df: pd.DataFrame) -> pd.DataFrame:
    """Nutritional status based on BMI"""
    if df['Nutritional Status'] is not None:
        print("Warning: Nutritional Status ALREADY PRESENT SKIPPING!")
        return df
    nutritional_status = pd.Series([])
    for i in range(len(df)):
        if df['BMI'][i] == 0.0:
            nutritional_status[i] = "NA"
        elif df['BMI'][i] < 18.5:
            nutritional_status[i] = "Underweight"
        elif df['BMI'][i] < 25:
            nutritional_status[i] = "Normal"
        elif 25 <= df['BMI'][i] < 30:
            nutritional_status[i] = "Overweight"
        elif df['BMI'][i] >= 30:
            nutritional_status[i] = "Obese"
        else:
            nutritional_status[i] = df['BMI'][i]
    df.insert(6, "Nutritional Status", nutritional_status)
    return df


def calculate_glucose_level(df: pd.DataFrame) -> pd.DataFrame:
    """Interpretation of OGTT (Glucose) - using OGTT levels recommended by DIABETES UK (2019)"""
    if df['Glucose'] is not None:
        print("Warning: Nutritional Status ALREADY PRESENT SKIPPING!")
        return df
    ogtt_interpretation = pd.Series([])
    for i in range(len(df)):
        if df['Glucose'][i] == 0.0:
            ogtt_interpretation[i] = "NA"

        elif df['Glucose'][i] <= 140:
            ogtt_interpretation[i] = "Normal"

        elif df['Glucose'][i] > 140 & df['Glucose'][i] <= 198:
            ogtt_interpretation[i] = "Impaired Glucose Tolerance"

        elif df['Glucose'][i] > 198:
            ogtt_interpretation[i] = "Diabetic Level"

        else:
            ogtt_interpretation[i] = df['Glucose'][i]
    df.insert(2, "Glucose Result", ogtt_interpretation)
    return df


def check_column_types(df):
    """
    Determines if each DataFrame column is discrete or continuous.

    Parameters:
    df (pandas.DataFrame): Input DataFrame

    Returns:
    dict: Mapping of column names to 'discrete' or 'continuous'
    """
    result = {}

    for column in df.columns:
        # Get column data type
        dtype = df[column].dtype

        # Check for object, string, or categorical types (typically discrete)
        if dtype == 'object' or dtype == 'string' or dtype.name == 'category':
            result[column] = 'discrete'
        # Check for numeric types
        elif np.issubdtype(dtype, np.number):
            # Count unique values
            unique_values = df[column].nunique()
            # If unique values are less than 1% of total or less than 20, consider discrete
            if unique_values < max(20, len(df) * 0.01):
                result[column] = 'discrete'
            else:
                result[column] = 'continuous'
        # Default to discrete for other types (like bool)
        else:
            result[column] = 'discrete'

    return result


def calculate_empty_data_in_a_feature(feature_name: str, df: pd.DataFrame) -> dict:
    """
    This function will be used to calculate the percent of missing data from the total set.
    It will ensure that Our Values matches with 'Table 2' of the Research Paper.
    Parameters:
        feature_name (str): Name of the feature
        df (pandas.DataFrame): Input DataFrame
    """

    # Calculate the number of zeros in SkinThickness
    num_zeros = (df[feature_name] == 0).sum()

    # Calculate the total number of rows
    total_rows = df.shape[0]

    # Calculate the percentage of zeros
    percentage_zeros = (num_zeros / total_rows) * 100
    return round(percentage_zeros, 1)
