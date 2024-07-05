import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AdamW
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from bert_model import BERTRegression, QADataset, collate_fn  # Import the model class

# Load your CSV file
df = pd.read_csv('patient_dataset.csv')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Check if a pretrained model exists
pretrained_model_path = 'bert_regression_model.pth'
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

try:
    # Load the pretrained model if it exists
    if os.path.exists(pretrained_model_path):
        model = BERTRegression()
        model.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        print(f"Pretrained model loaded from {pretrained_model_path}")
    else:
        model = BERTRegression()
        print(f"No pretrained model found at {pretrained_model_path}. Initializing new model.")
except Exception as e:
    print(f"Error loading pretrained model: {e}")
    model = BERTRegression()

model.to(device)  # Move model to GPU if available

optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = nn.MSELoss()

# Training setup
dataset = QADataset(df)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# Training loop
model.train()
for epoch in range(20):  # Replace with desired number of epochs
    for input_ids, attention_mask, score in dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, score = input_ids.to(device), attention_mask.to(device), score.to(device)
        output = model(input_ids, attention_mask)
        loss = criterion(output.squeeze(), score)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'bert_regression_model.pth')
print("Model trained and saved to disk.")
