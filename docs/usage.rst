=====
Usage
=====

To use CririsSum in a project::

    import cririssum

Data Gathering
==============

The project includes a Jupyter notebook for data collection and preprocessing, located in the `data/` directory.

**File:** `data_collection.ipynb`

**Purpose:**  
This notebook is used to gather and preprocess data for the CRISISFacts project. It automates the downloading and preparation of datasets required for analysis.

**Steps to Use:**
1. Navigate to the `data/` directory in your terminal or file explorer.
2. Open the notebook using Jupyter Notebook or Jupyter Lab
    ```
    jupyter notebook data_collection.ipynb
    ```
3. Follow the instructions provided within the notebook to execute the data gathering process.

**Dependencies:**  
Ensure that the necessary Python libraries and tools are installed by referring to the `requirements_dev.txt` file.

**Note:**  
Once the data is collected and preprocessed using this notebook, you can use the resulting datasets with the `cririssum` package for summarization tasks.