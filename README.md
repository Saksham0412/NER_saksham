# NER_saksham
This project demonstrates how to download, preprocess, and use the CoNLL-2003 dataset for Named Entity Recognition (NER) with BERT from Google Research.

Overview
Named Entity Recognition (NER) is a common NLP task where the objective is to classify tokens in a text as names of real-world entities, such as people, organizations, and locations. In this project, we use BERT, a powerful transformer-based model, to perform NER on the CoNLL-2003 dataset.

Dataset
The CoNLL-2003 dataset is widely used for NER tasks and includes labeled entities in four categories:

PER: Person
ORG: Organization
LOC: Location
MISC: Miscellaneous
The dataset is formatted in multiple files (e.g., eng.train, eng.testa, eng.testb), each containing sentences with token, POS, chunk, and entity tags.

Dataset Download Link
The CoNLL-2003 dataset can be accessed and downloaded here.

Setup and Requirements
Prerequisites
Python 3.6 or higher
transformers library for BERT
pandas for data manipulation
sklearn for data splitting
requests and zipfile for downloading and extracting data
