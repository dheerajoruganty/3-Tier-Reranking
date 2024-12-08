from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType, ArrayType
import spacy

def initialize_spark():
    
    """
    Initializes a Spark session for data processing.

    Returns:
        SparkSession: An active Spark session.
    """
    return SparkSession.builder.appName("CRISISFacts Preprocessing").getOrCreate()

def tokenize_and_remove_stopwords(text, nlp):
    """
    Tokenizes input text and removes stopwords.

    Parameters:
        text (str): The input text to process.
        nlp (spacy.lang): A SpaCy language model for tokenization.

    Returns:
        str: Processed text with stopwords removed.
    """
    if not text:
        return ""
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def recognize_and_normalize_entities(text, nlp):
    """
    Performs entity recognition and normalizes entities.

    Parameters:
        text (str): The input text to process.
        nlp (spacy.lang): A SpaCy language model for entity recognition.

    Returns:
        str: Text with normalized named entities.
    """
    if not text:
        return ""
    doc = nlp(text)
    entities = {ent.text: ent.label_ for ent in doc.ents}
    return " | ".join(f"{text}:{label}" for text, label in entities.items())

def preprocess_pipeline(input_parquet, output_path):
    """
    Preprocesses the input data pipeline with tokenization, stopword removal, and entity recognition.

    Parameters:
        input_parquet (str): Path to the input Parquet file.
        output_path (str): Directory to save the preprocessed data.
    """
    # Initialize Spark session
    spark = initialize_spark()

    # Load Parquet data
    df = spark.read.parquet(input_parquet)

    # Initialize SpaCy language model
    nlp = spacy.load("en_core_web_sm")

    # Define UDFs for tokenization and entity recognition
    tokenize_udf = udf(lambda text: tokenize_and_remove_stopwords(text, nlp), StringType())
    entity_udf = udf(lambda text: recognize_and_normalize_entities(text, nlp), StringType())

    # Apply UDFs for preprocessing
    preprocessed_df = df.withColumn("cleaned_text", tokenize_udf(col("text"))) \
                        .withColumn("entities", entity_udf(col("text")))

    # Show sample output
    preprocessed_df.show(truncate=False)

    # Save preprocessed data
    preprocessed_df.write.mode("overwrite").parquet(f"{output_path}/preprocessed_data.parquet")
    preprocessed_df.write.mode("overwrite").csv(f"{output_path}/preprocessed_data.csv", header=True)

if __name__ == "__main__":
    # File paths
    input_parquet_path = "data/cleaned_crisisfacts_data.parquet"
    output_directory = "data/preprocessed_crisisfacts"

    # Run preprocessing pipeline
    preprocess_pipeline(input_parquet_path, output_directory)
