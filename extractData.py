import pandas as pd
from io import StringIO

def extract_arff_data(file_path):
    """
    Extracts and processes the @DATA section of an ARFF file.
    
    Parameters:
        file_path (str): Path to the ARFF file.
    
    Returns:
        pd.DataFrame: Extracted data as a Pandas DataFrame.
    """
    # Step 1: Read the raw file
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    # Step 2: Locate the @DATA section
    try:
        data_start = lines.index('@DATA\n') + 1
    except ValueError:
        raise ValueError("The file does not contain a @DATA section.")
    
    # Step 3: Extract column names
    attributes = [line for line in lines if line.startswith('@ATTRIBUTE')]
    column_names = [attr.split()[1] for attr in attributes]
    
    # Step 4: Extract the data lines
    data_lines = lines[data_start:]
    data_string = ''.join(data_lines)  # Combine lines into a single CSV string
    
    # Step 5: Convert to a DataFrame
    data_df = pd.read_csv(StringIO(data_string), header=None)
    data_df.columns = column_names  # Assign column names
    
    return data_df

# Usage example
file_path = "/Users/hetavpatel/Desktop/Data Science/Grad DS Work/DSCI 789 Explainable AI/Assignment 5/datasets/Bank_Market_Dataset"  # Replace with your file path
try:
    extracted_data = extract_arff_data(file_path)
    print(extracted_data.head())  # Display the first few rows
    #save the extracted data to a csv file
    extracted_data.to_csv("Bank_Market_Dataset.csv")
except Exception as e:
    print(f"Error: {e}")