import pandas as pd
import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
train_df = pd.read_excel("train.xlsx")
dev_df = pd.read_excel("dev.xlsx")

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.strip()  # Remove extra spaces
    return text

# Apply preprocessing
train_df["text"] = train_df["text"].astype(str).apply(preprocess_text)
dev_df["text"] = dev_df["text"].astype(str).apply(preprocess_text)

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

# Load BERT model
model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define a custom dataset class
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # Increased max length
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        encoding['labels'] = torch.tensor(label, dtype=torch.long)
        return encoding

# Prepare training and validation datasets
train_dataset = HateSpeechDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
dev_dataset = HateSpeechDataset(dev_df["text"].tolist(), dev_df["label"].tolist(), tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increased batch size
dev_dataloader = DataLoader(dev_dataset, batch_size=32)

# Setup optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)  # Added weight decay
num_epochs = 3  
total_steps = len(train_dataloader) * num_epochs

scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps=0, 
                                            num_training_steps=total_steps)

# Training loop with progress printing
for epoch in range(num_epochs):  
    model.train()
    print(f"Epoch {epoch+1}/{num_epochs} - Training in progress...")
    for batch_idx, batch in enumerate(train_dataloader):
        optimizer.zero_grad()
        
        # Move tensors to the selected device
        input_ids = batch['input_ids'].squeeze(1).to(device)
        attention_mask = batch['attention_mask'].squeeze(1).to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        output = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = output.loss
        loss.backward()
        optimizer.step()
        scheduler.step()  # Update learning rate

        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1}/{len(train_dataloader)} batches...")

    # Evaluate after each epoch
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        print("Evaluating on dev dataset...")
        for batch_idx, batch in enumerate(dev_dataloader):
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            labels = batch['labels'].to(device)

            output = model(input_ids, attention_mask=attention_mask)
            logits = output.logits
            preds = torch.argmax(logits, dim=1).cpu().tolist()

            predictions.extend(preds)
            true_labels.extend(labels.cpu().tolist())

            # Print progress every 100 batches
            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {batch_idx + 1}/{len(dev_dataloader)} batches...")

    # Calculate accuracy and classification report
    acc = accuracy_score(true_labels, predictions)
    print(f"Epoch {epoch+1} - Accuracy: {acc}")
    print(classification_report(true_labels, predictions))

# Function to predict labels for test data
def predict_from_file(input_file, output_file):
    test_df = pd.read_excel(input_file)
    test_df["text"] = test_df["text"].astype(str).apply(preprocess_text)
    
    predictions = []
    for idx, text in enumerate(test_df["text"].tolist()):
        encoding = tokenizer(text, truncation=True, padding='max_length', max_length=256, return_tensors='pt').to(device)  # Increased max length
        with torch.no_grad():
            output = model(**encoding)
            prediction = torch.argmax(output.logits, dim=1).cpu().item()
        predictions.append(prediction)

        # Print every 50th prediction to track progress
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(test_df)} samples...")

    # Save predictions to CSV without header
    test_df["predictions"] = predictions
    test_df[["id", "predictions"]].to_csv(output_file, index=False, header=False)
    print(f"Predictions saved to {output_file}")

# Example Usage
predict_from_file("test.xlsx", "output.csv")
