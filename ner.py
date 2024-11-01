import os
import requests
import zipfile
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

# Define URLs and paths
url = "https://data.deepai.org/conll2003.zip"  # Replace with the appropriate download link if needed
download_path = "conll2003.zip"
extract_path = "CoNLL2003"
data_files = ["eng.train", "eng.testa", "eng.testb"]

# Step 1: Download the dataset
if not os.path.exists(download_path):
    print("Downloading CoNLL-2003 dataset...")
    response = requests.get(url)
    with open(download_path, "wb") as f:
        f.write(response.content)
    print("Download complete.")

# Step 2: Extract the dataset
if not os.path.exists(extract_path):
    print("Extracting dataset...")
    with zipfile.ZipFile(download_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

# Initialize lists to store sentences and labels
sentences, labels = [], []
sentence, label = [], []

# Step 3: Load and process data from each file
for data_file in data_files:
    file_path = os.path.join(extract_path, data_file)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    print(f"Processing file: {data_file}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                # End of a sentence
                if sentence:
                    sentences.append(sentence)
                    labels.append(label)
                    sentence, label = [], []
            else:
                parts = line.strip().split()
                token, entity_label = parts[0], parts[-1]  # Assuming format: Token POS CHUNK ENTITY
                sentence.append(token)
                label.append(entity_label)

# Step 4: Create DataFrame
conll_df = pd.DataFrame({
    "Sentence": [" ".join(sent) for sent in sentences],
    "Labels": [" ".join(lbl) for lbl in labels]
})

# Step 5: Split data into training and test sets
train_df, test_df = train_test_split(conll_df, test_size=0.2, random_state=42)

# Step 6: Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenization function
def preprocess_for_bert(data, tokenizer, max_length=128):
    """Tokenizes and preprocesses text for BERT."""
    tokens = tokenizer(
        data["Sentence"].tolist(),
        max_length=max_length,
        padding="max_length",
        truncation=True,
        is_split_into_words=True,
        return_tensors="pt"  # Returns tensors for PyTorch
    )
    return tokens

# Apply tokenization to train and test sets
train_tokens = preprocess_for_bert(train_df, tokenizer)
test_tokens = preprocess_for_bert(test_df, tokenizer)

# Print a sample of tokenized inputs
print("Sample tokenized input for training set:")
print(train_tokens.input_ids[0])  # Display token IDs for first entry in the training set
print("Corresponding labels:", train_df.iloc[0]["Labels"])

# Optional: Save processed data
train_df.to_csv("train_conll2003.csv", index=False)
test_df.to_csv("test_conll2003.csv", index=False)

print("Data processing complete. Ready for BERT model training.")
