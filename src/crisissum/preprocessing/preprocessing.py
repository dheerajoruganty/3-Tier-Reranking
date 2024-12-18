import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import argparse

# Download NLTK data files
nltk.download("punkt")
nltk.download("stopwords")


def load_parquet(file_path):
    """
    Loads a Parquet file into a Pandas DataFrame.

    Parameters:
        file_path (str): Path to the Parquet file.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    print(f"Loading data from {file_path}...")
    try:
        return pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        raise


def preprocess_text(df, text_column):
    """
    Tokenizes text, removes stopwords, and performs entity recognition and normalization.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column containing text data.

    Returns:
        pd.DataFrame: DataFrame with preprocessed text and recognized entities.
    """
    print("Preprocessing text data...")
    nlp = spacy.load("en_core_web_sm")
    stop_words = set(stopwords.words("english"))

    def process_text(text):
        """
        Tokenizes text, removes stopwords, performs entity recognition, and normalizes entities.

        Parameters:
            text (str): Input text to preprocess.

        Returns:
            tuple: Preprocessed text (str) and recognized entities (list).
        """
        try:
            # Tokenization and stopword removal
            tokens = word_tokenize(text)
            filtered_tokens = [
                word for word in tokens if word.lower() not in stop_words
            ]
            preprocessed_text = " ".join(filtered_tokens)

            # Entity recognition
            doc = nlp(preprocessed_text)
            entities = [(ent.text, ent.label_) for ent in doc.ents]

            return preprocessed_text, entities
        except Exception as e:
            print(f"Error processing text: {e}")
            return "", []

    # Apply the process_text function to the text column
    df["preprocessed_text"], df["entities"] = zip(*df[text_column].apply(process_text))
    print("Text preprocessing completed.")
    return df


def save_preprocessed_data(df, output_path):
    """
    Saves the preprocessed DataFrame to a file.

    Parameters:
        df (pd.DataFrame): DataFrame to save.
        output_path (str): File path to save the DataFrame.

    Returns:
        None
    """
    try:
        print(f"Saving preprocessed data to {output_path}...")
        df.to_csv(output_path, index=False)
        print(f"Data saved successfully to {output_path}.")
    except Exception as e:
        print(f"Error saving data: {e}")
        raise


def main():
    """
    Main function for preprocessing text data.
    """
    # Argument parser
    parser = argparse.ArgumentParser(description="Text Preprocessing Pipeline")
    parser.add_argument(
        "--input_file", type=str, required=True, help="Path to the input Parquet file."
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of the text column in the dataset.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the preprocessed CSV file.",
    )
    args = parser.parse_args()

    # Load data
    data = load_parquet(args.input_file)

    # Preprocess text
    preprocessed_data = preprocess_text(data, text_column=args.text_column)

    # Save preprocessed data
    save_preprocessed_data(preprocessed_data, args.output_file)


if __name__ == "__main__":
    main()
