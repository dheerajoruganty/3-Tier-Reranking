import pandas as pd
import os
import csv

# Paths
data_dir = "../../../data/cleaned_crisisfacts_data.csv"
output_file = "../../../data/combined_data.csv"


def combine_csv_files(input_dir, output_file):
    """Combine multiple CSV files into one."""
    all_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")
    ]
    combined_dataframes = []

    for f in all_files:
        try:
            # Infer delimiter and read CSV
            with open(f, "r") as file:
                dialect = csv.Sniffer().sniff(file.read(1024))
                file.seek(0)
            df = pd.read_csv(
                f,
                delimiter=dialect.delimiter,
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",
            )

            # Standardize columns
            expected_columns = [
                "doc_id",
                "event",
                "text",
                "source",
                "source_type",
                "unix_timestamp",
            ]
            df = df.reindex(columns=expected_columns, fill_value=None)

            # Drop rows with excessive missing data
            df = df.dropna(thresh=3)  # Keep rows with at least 3 non-NaN values

            combined_dataframes.append(df)
            print(f"Processed {f} successfully.")
        except Exception as e:
            print(f"Error processing {f}: {e}")

    # Combine all DataFrames
    if combined_dataframes:
        combined_df = pd.concat(combined_dataframes, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined {len(all_files)} files into {output_file}.")
    else:
        print("No valid files to combine.")


if __name__ == "__main__":
    combine_csv_files(data_dir, output_file)
