import numpy as np
import pandas as pd

def prepare_data(df):
    conditions = [
        (df['category'] == 'color') & (df['shape_color'] == 'white'),
        (df['category'] == 'color') & (df['shape_color'] == 'colorful'),
        (df['category'] == 'shape') & (df['shape_type'] == 'square'),
        (df['category'] == 'shape') & (df['shape_type'] == 'circle')
    ]

    # Create a list of the values we want to assign for each condition
    values = ['over', 'under', 'over', 'under']

    # create a new column and use np.select to assign values to it using our lists as arguments
    df['group'] = np.select(conditions, values)

    # # Add column that contains the prediction difference
    df["prediction_diff"] = df['prediction'] - df['actual']
    
    # Add column for error (RMSE)
    df["rmse"] = np.sqrt(df["prediction_diff"] * df["prediction_diff"])

# Function to load data
def load_run_data(run, date, results_path):
    # Load results file from each run into one large pandas dataframe
    excel_name = "{}_{}_results.xlsx".format(run, date)
    df = pd.read_excel(results_path + excel_name)
    prepare_data(df)
    return df