import torch
from transformers import BertTokenizer
from bert_model import BERTRegression  # Import the model class

# Load the model
model = BERTRegression()
model.load_state_dict(torch.load('bert_regression_model.pth'))
model.eval()

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Test examples
test_examples = [
    ("How much do your legs hurt right now?", "I feel a throbbing sensation in both legs, worse in the right."),
    ("Describe the pain in your legs.", "The pain is like a dull ache, not too intense but persistent."),
    ("Do you feel any discomfort in your legs?", "Yes, there's a mild throbbing in my left leg."),
    ("What kind of pain are you experiencing?", "It's a sharp pain that radiates down my right leg."),
    ("Rate the intensity of your leg pain on a scale of 1 to 10.", "It's about a 7, quite intense."),
    ("How are your legs feeling today?", "They feel stiff and sore, especially after exercise."),
    ("Are you able to walk comfortably?", "No, the pain makes walking difficult."),
    ("Have you tried any pain relief methods?", "Yes, I've used ice packs which provide some relief."),
    ("Is the pain constant or does it come and go?", "It comes and goes, but it's persistent."),
    ("Does the pain affect your daily activities?", "Yes, it limits my ability to stand for long periods."),
    ("When did you first notice the pain?", "It started a few days ago after a long hike."),
]

# Output file path
output_file = "predicted_scores.txt"

# Evaluate each test example and write to file
with open(output_file, 'w') as f:
    for i, (new_question, new_answer) in enumerate(test_examples):
        inputs = tokenizer(new_question, new_answer, return_tensors='pt', padding=True, truncation=True)
        with torch.no_grad():
            predicted_score = model(inputs['input_ids'], inputs['attention_mask'])
        f.write(f"Test Example {i + 1}:\n")
        f.write(f"Question: {new_question}\n")
        f.write(f"Answer: {new_answer}\n")
        f.write(f"Predicted Score: {predicted_score.item():.4f}\n")
        f.write("=" * 50 + "\n")

print(f"Output saved to {output_file}")