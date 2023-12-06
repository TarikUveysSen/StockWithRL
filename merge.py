import pandas as pd

def join_dataframes(feature_dict, ticker_dict):
    result_df = None

    for key, columns_to_use in ticker_dict.items():
        # Check if the key is present in the feature dictionary
        if key in feature_dict:
            # Copy the DataFrame from the feature dictionary
            df = feature_dict[key].copy()

            # Check if columns_to_use is a list of valid column indices
            if isinstance(columns_to_use, list) and all(isinstance(idx, int) for idx in columns_to_use):
                # Select only the specified columns from the DataFrame using iloc
                df_selected = df.iloc[:, columns_to_use]

                # Perform the left join operation with indices
                if result_df is None:
                    result_df = df_selected
                else:
                    result_df = pd.merge(result_df, df_selected, how='left', left_index=True, right_index=True)
            else:
                print(f"Invalid columns_to_use for key '{key}'. Please provide a list of column indices.")

        else:
            print(f"Key '{key}' not found in feature_dict.")

    return result_df
