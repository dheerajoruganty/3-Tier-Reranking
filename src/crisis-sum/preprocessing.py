import pandas as pd
import spacy
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data files
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

def load_parquet(file_path):
    """
    Loads a Parquet file into a Pandas DataFrame.

    Parameters:
        file_path (str): Path to the Parquet file.
    
    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    return pd.read_parquet(file_path)

def preprocess_text(df, text_column):
    """
    Tokenizes text, removes stopwords, and performs entity recognition and normalization.

    Parameters:
        df (pd.DataFrame): DataFrame containing the text data.
        text_column (str): Name of the column containing text data.
    
    Returns:
        pd.DataFrame: DataFrame with preprocessed text and recognized entities.
    """
    # Load NLP models
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
        # Tokenization and stopword removal
        tokens = word_tokenize(text)
        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        preprocessed_text = " ".join(filtered_tokens)

        # Entity recognition
        doc = nlp(preprocessed_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]  # Extract entities and their labels

        return preprocessed_text, entities

    # Apply the process_text function to the text column
    df["preprocessed_text"], df["entities"] = zip(*df[text_column].apply(process_text))
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
    df.to_csv(output_path, index=False)

# Main pipeline
if __name__ == "__main__":
    # Step 1: Load data
    input_file = "data/cleaned_crisisfacts_data.parquet"
    data = load_parquet(input_file)

    # Step 2: Preprocess data
    preprocessed_data = preprocess_text(data, text_column="text")

    # Step 3: Save preprocessed data
    output_file = "data/preprocessed_crisisfacts_data.csv"
    save_preprocessed_data(preprocessed_data, output_file)

    print(f"Preprocessed data saved to {output_file}.")
