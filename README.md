# People Wikipedia Clustering Project

## Overview

This project is the final submission for the ML2 (Unsupervised Learning) course. The goal of this project is to perform clustering on textual data extracted from Wikipedia articles about various people. We utilize Natural Language Processing (NLP) techniques to preprocess the text data and apply KMeans clustering to identify groups of similar articles.

## Dataset

The dataset used in this project is the **People Wikipedia** dataset, which extracts structured content from Wikipedia. You can find the dataset [here](https://www.kaggle.com/datasets/sameersmahajan/people-wikipedia-data).

### Data Description

The dataset contains articles about various individuals, with each article represented as a text entry. The primary column of interest is `text`, which contains the content of the Wikipedia articles.

## Project Structure

The project consists of the following main components:

1. **Preprocessing**: The `preprocessing.py` file contains functions for text preprocessing, including:
   - Lowercasing the text
   - Removing URLs, email addresses, special characters, and numbers
   - Tokenization
   - Stopword removal
   - Lemmatization

   The `tfidf_scaled` function computes the TF-IDF representation of the preprocessed text and scales the features.

2. **Visualization**: The `visualization.py` file contains functions for visualizing the clustering results using PCA (Principal Component Analysis):
   - `Two_D_plot`: Creates a 2D scatter plot of the clustered data.
   - `three_D_plot`: Creates a 3D scatter plot of the clustered data.
   - `three_D_interactive`: Creates an interactive 3D scatter plot using Plotly.

3. **Main Execution**: The `main.py` file orchestrates the entire process:
   - It reads the dataset.
   - It sets up a scikit-learn pipeline that includes preprocessing, TF-IDF scaling, and KMeans clustering.
   - It visualizes the clustering results using the functions from the visualization module.

## Pipeline

The project utilizes a scikit-learn `Pipeline` to streamline the workflow. The pipeline consists of the following steps:

1. **Text Preprocessing**: The `FunctionTransformer` applies the `preprocess_text_simple` function to each text entry, cleaning and preparing the data for analysis.

2. **TF-IDF Scaling**: Another `FunctionTransformer` applies the `tfidf_scaled` function to convert the preprocessed text into a TF-IDF representation and scales the features.

3. **KMeans Clustering**: The KMeans algorithm is applied to the TF-IDF features to identify clusters of similar articles. The number of clusters is set to 3.

After fitting the pipeline, the cluster labels are obtained, and the results are visualized using 2D and 3D plots.

## Requirements

To run this project, you will need the following Python packages:

- pandas
- scikit-learn
- nltk
- matplotlib
- plotly

You can install the required packages using pip:

```bash
pip install pandas scikit-learn nltk matplotlib plotly
