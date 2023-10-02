"""
This file contains code to generate the csv file with the desired number for each targeted group. 
This generated csv file can then be used to be used as training set in the generate_trained_model.ipynb file.
"""

import pandas as pd
import numpy as np

# Typically I use data_name to signify which data I used, like 'black' means the file black_dataset.csv, 'balanced' means the file balanced_dataset.csv
#data_name = 'black' # 9000 black, 500 women, 200 trans, 150 gay, 150 lesbian
#data_name = 'women' # 9000 women, 500 black, 200 trans, 150 gay, 150 lesbian
data_name = 'balanced' # 3300 black, 3300 women, 2800 trans, 100 gay, 500 lesbian

hate_score_threshold = 0.5 # Score above this is considered hate speech, lower this threshold if you want more speech to be classed hate speech
num_targeted_asian = 0 # Total asian targeted is 7025, try not to set this higher than this number
num_targeted_black = 3300 # Max 22899
num_targeted_white = 0 # Max 9797
num_targeted_woman = 3300 # Max 27889
num_targeted_transgender = 2800 # 3326 + 4703 + 2611, combined transgender men + unspecified + women, there may be overlap
num_targeted_gay = 100 # Max 15465
num_targeted_lesbian = 500 # Max 6883
num_targeted_others = 0

def split_csv(source_csv, destination_csv):
    # Read the CSV file
    df = pd.read_csv(source_csv)

    # Start with an empty DataFrame for results
    result = pd.DataFrame()

    print(f"Number of rows in the result: {len(result)}")

    # Asian is column 22
    condition_asian = df[df.columns[22]]
    filtered_asian = df[condition_asian].head(num_targeted_asian)
    df.drop(filtered_asian.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_asian]) 
    print(f"Number of rows in the result: {len(result)}")

    # Black is column 23
    condition_black = df[df.columns[23]]
    filtered_black = df[condition_black].head(num_targeted_black)
    df.drop(filtered_black.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_black]) 
    print(f"Number of rows in the result: {len(result)}")

    # White is column 28
    condition_white = df[df.columns[28]]
    filtered_white = df[condition_white].head(num_targeted_white)
    df.drop(filtered_white.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_white]) 
    print(f"Number of rows in the result: {len(result)}")

    # Woman is column 51
    condition_woman = df[df.columns[51]]
    filtered_woman = df[condition_woman].head(num_targeted_woman)
    df.drop(filtered_woman.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_woman]) 
    print(f"Number of rows in the result: {len(result)}")

    # Trans is column 48 to 50
    condition_transgender = df[df.columns[48:51]].any(axis=1)
    filtered_transgender = df[condition_transgender].head(num_targeted_transgender)
    df.drop(filtered_transgender.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_transgender]) 
    print(f"Number of rows in the result: {len(result)}")

    # Gay is column 55
    condition_gay = df[df.columns[55]]
    filtered_gay = df[condition_gay].head(num_targeted_gay)
    df.drop(filtered_gay.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_gay]) 
    print(f"Number of rows in the result: {len(result)}")

    # Lesbian is column 56
    condition_lesbian = df[df.columns[56]]
    filtered_lesbian = df[condition_lesbian].head(num_targeted_lesbian)
    df.drop(filtered_lesbian.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_lesbian])
    print(f"Number of rows in the result: {len(result)}")

    # For other targeted not specified above
    condition_others = ~(condition_asian | condition_black | condition_white | condition_woman | condition_transgender | condition_gay | condition_lesbian)
    filtered_others = df[condition_others].head(num_targeted_others)
    df.drop(filtered_others.index, inplace=True)  # Remove these rows from the main df
    result = pd.concat([result, filtered_others]) 
    print(f"Number of rows in the result: {len(result)}")

    # Reduce to just Class and Tweet column, and a few targted groups that we need
    result['class'] = np.where(result.iloc[:, 13] < hate_score_threshold, 2, 0) # 0 for hate speech and 2 if not
    result = result[['class', df.columns[14], df.columns[23], df.columns[48], df.columns[49], df.columns[50], df.columns[51], df.columns[55], df.columns[56]]]  # The "Tweet" column
    result.columns = ["class", "text", "black", "trans_men", "trans_unspecified", "trans_women", "women", "gay", "lesbian"]

    # Shuffle the rows so they will be in random order
    result = result.sample(frac=1).reset_index(drop=True)

    # Write to a new CSV file
    result.to_csv(destination_csv, index=False)
    print("Finish")

# Example usage
source_csv_path = "berkeley_speech_dataset.csv"
destination_csv_path = f'{data_name}_dataset.csv'
split_csv(source_csv_path, destination_csv_path)
