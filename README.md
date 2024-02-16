# Text-RGNNs: Relational Modeling for Heterogeneous Text Graphs

This repository hosts the code accompanying the paper submitted to IEEE SPL.

## How to Run?

Please address each section below, we gave them in specific order. Follow each step.

### Datasets

Before running the code, ensure you have the necessary datasets. We utilized MR, R8, R52, and Ohsumed datasets, commonly employed for sentiment classification in the graph domain. Download these datasets from the provided link. Once downloaded, create a folder named "data" and place the corpus under "data/original-data".

For datasets sourced from GLUE, such as SST-2 and CoLA, there's no need to download them separately.

### Preliminaries

Under the src directory, `__init__.py` file is used to solve the path issues. Change the `PROJECT_PATH` variable accordingly. To prepare the required Word2Vec embeddings and TF-IDF graphs, run the `generator.py` script. These embeddings and graphs are essential for constructing the text graph structure.
