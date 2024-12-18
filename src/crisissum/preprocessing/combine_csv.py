import pandas as pd
import os
import csv
import argparse


def combine_csv_files(input_dir: str, output_file: str) -> None:
    """
    Combine multiple CSV files from a directory into a single CSV file.

    This function reads all CSV files in the specified directory, standardizes
    columns, handles bad lines, removes rows with excessive missing values,
    and combines all data into a single CSV file.

    Args:
        input_dir (str): Path to the directory containing input CSV files.
        output_file (str): Path to the output combined CSV file.

    Raises:
        FileNotFoundError: If the input directory does not exist or is empty.
        Exception: For any issues encountered while reading or combining files.

    Example:
        combine_csv_files("data/input", "data/combined_output.csv")
    """
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    # Identify all CSV files in the input directory
    all_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")
    ]

    if not all_files:
        raise FileNotFoundError(f"No CSV files found in directory: {input_dir}")

    # List to store DataFrames from each file
    combined_dataframes = []

    # Expected column names for consistency
    expected_columns = [
        "doc_id",
        "event",
        "text",
        "source",
        "source_type",
        "unix_timestamp",
    ]

    # Iterate over all CSV files
    for file_path in all_files:
        try:
            # Infer delimiter automatically
            with open(file_path, "r") as file:
                sample_data = file.read(1024)
                dialect = csv.Sniffer().sniff(sample_data)
                file.seek(0)

            # Read CSV with inferred delimiter
            df = pd.read_csv(
                file_path,
                delimiter=dialect.delimiter,
                quotechar='"',
                escapechar="\\",
                on_bad_lines="skip",  # Skip badly formatted lines
            )

            # Reindex to standardize columns and fill missing ones with None
            df = df.reindex(columns=expected_columns, fill_value=None)

            # Drop rows with less than 3 valid values
            df = df.dropna(thresh=3)

            combined_dataframes.append(df)
            print(f"Successfully processed file: {file_path}")

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # Combine all DataFrames and save to the output file
    if combined_dataframes:
        combined_df = pd.concat(combined_dataframes, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined {len(all_files)} files into '{output_file}'.")
    else:
        print("No valid files found to combine.")


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Combine CSV Files")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Path to the input directory containing CSV files.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the combined CSV file.",
    )
    args = parser.parse_args()

    try:
        combine_csv_files(args.input_dir, args.output_file)
    except FileNotFoundError as fnf_error:
        print(f"[Error]: {fnf_error}")
    except Exception as e:
        print(f"[Unexpected Error]: {e}")
