import pandas as pd

def join_dataframes(ticker_dict, feature_dict):
    result_df = None

    for key, df in ticker_dict.items():
        # Check if the key is present in the feature dictionary
        if key in feature_dict:
            # Copy the DataFrame from the ticker dictionary
            df_copy = df.copy()

            # Select only the specified columns from the DataFrame using column names
            columns_to_use = feature_dict[key]
            if isinstance(columns_to_use, list) and all(isinstance(col, str) for col in columns_to_use):
                df_selected = df_copy[columns_to_use]

                # Perform the left join operation with indices
                if result_df is None:
                    result_df = df_selected
                else:
                    result_df = pd.merge(result_df, df_selected, how='left', left_index=True, right_index=True)
            else:
                print(f"Invalid columns_to_use for key '{key}'. Please provide a list of column names.")
        else:
            print(f"Key '{key}' not found in feature_dict.")

    return result_df
