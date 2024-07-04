from transformers import BertTokenizer, BertForSequenceClassification
import torch

def classifier_scibert(sentence):
    output_dir = './trained_model_to_send/'
    model = BertForSequenceClassification.from_pretrained(output_dir)
    tokenizer = BertTokenizer.from_pretrained(output_dir)

    # Tokenize input sentences
    tokenized_input = tokenizer(sentence, padding=True, truncation=True, return_tensors="pt")

    # Ensure the model is in evaluation mode
    model.eval()

    # Make predictions
    with torch.no_grad():
        # Forward pass
        outputs = model(**tokenized_input)

    # Get the predicted probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

    # Get the predicted class (0 or 1 in binary classification, 0 == No N-Value, 1 == N-Value)
    predicted_class = torch.argmax(probs, dim=1).tolist()

    # Display results
    #print(f"Sentence: {sentence}")
    #print(f"Predicted Label: {predicted_class[0]}")
    #print()
    return predicted_class[0]

classification = classifier_scibert('There were no significant differences between the two groups')
print(classification)