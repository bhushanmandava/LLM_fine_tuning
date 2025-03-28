##  Movie Sentiment Analysis with DistilBERT and LoRA

### Overview

This notebook demonstrates fine-tuning a pre-trained DistilBERT model for sentiment analysis on movie reviews. It uses Parameter-Efficient Fine-Tuning (PEFT) via Low-Rank Adaptation (LoRA) to achieve good performance while training a small fraction of the model's parameters. This approach is particularly useful when computational resources are limited.

### 1. Setup and Imports

```python
from datasets import load_dataset, DatasetDict, Dataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
import evaluate
import torch
import numpy as np
```

**Explanation:**

*   **`datasets`**:  Hugging Face's library for easily loading and manipulating datasets.  We'll use it to load the IMDB dataset.  `DatasetDict` and `Dataset` are the core data structures.
*   **`transformers`**:  The core library for using pre-trained models.  Key components:
    *   `AutoTokenizer`:  Automatically loads the appropriate tokenizer for the chosen model.
    *   `AutoConfig`:  Automatically loads the configuration for the model.
    *   `AutoModelForSequenceClassification`:  Loads a pre-trained model with a sequence classification head (for sentiment analysis).
    *   `DataCollatorWithPadding`:  A utility for padding sequences to the same length within a batch during training.
    *   `TrainingArguments`:  Specifies training parameters (learning rate, batch size, etc.).
    *   `Trainer`:  A high-level class that simplifies the training loop.
*   **`peft`**:  The library for Parameter-Efficient Fine-Tuning.  Specifically, we're using LoRA:
    *   `LoraConfig`:  Defines the LoRA configuration (rank, scaling, etc.).
    *   `get_peft_model`:  Applies the LoRA adapters to the base model.
*   **`evaluate`**: Hugging Face's library for calculating metrics.
*   **`torch`**: PyTorch, the deep learning framework.
*   **`numpy`**: Numerical computing library.

### 2. Model and Tokenizer Initialization

```python
model_checkpoint = 'distilbert-base-uncased'

id2label = {
    0: 'Negative',
    1: 'Positive'
}
label2id = {"Negative": 0, "Positive": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    model_checkpoint, num_labels=2, id2label=id2label, label2id=label2id
)

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, add_prefix_space=True)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': ''})
    model.resize_token_embeddings(len(tokenizer))
```

**Explanation:**

*   **`model_checkpoint`**:  Specifies the pre-trained model to use (DistilBERT in this case).  "distilbert-base-uncased" is a good balance of size and performance.
*   **`id2label` / `label2id`**:  Dictionaries mapping integer IDs to string labels (and vice-versa) for the sentiment classes.
*   **`AutoModelForSequenceClassification.from_pretrained(...)`**:  Loads the pre-trained DistilBERT model and adds a classification head on top.  `num_labels=2` indicates a binary classification problem (positive/negative).  The `id2label` and `label2id` mappings are passed to the model for convenience.
*   **`AutoTokenizer.from_pretrained(...)`**:  Loads the tokenizer associated with the DistilBERT model. `add_prefix_space=True` is important for some tokenizers to ensure correct tokenization of the first word in a sequence.
*   **Padding:**  The code checks if the tokenizer has a padding token. If not, it adds a `` token and resizes the model's embedding layer to accommodate the new token. This is crucial for batching sequences of different lengths.

### 3. Dataset Loading and Preprocessing

```python
def tokinizer_function(examples):
    text = examples["text"]
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="np",
        truncation=True,
        max_length=512
    )
    return tokenized_inputs
```

**Explanation:**

*   **`tokinizer_function(examples)`**: This function takes a batch of examples (movie reviews) and tokenizes them.
    *   `tokenizer.truncation_side = "left"`:  Truncates the beginning of the sequence if it exceeds `max_length`. This is often a good strategy for sentiment analysis, as the ending of a review might contain the most important sentiment information.
    *   `tokenizer(...)`:  The core tokenization step.  It converts the text into a sequence of token IDs, adds special tokens (like $$CLS] and $$SEP]), and truncates/pads the sequence.
        *   `return_tensors="np"`:  Returns NumPy arrays, which are then converted to PyTorch tensors later.
        *   `truncation=True`:  Enables truncation to `max_length`.
        *   `max_length=512`:  The maximum length of the tokenized sequence.

### 4. Metric Definition

```python
def compute_metrics(p):
    predictions, labels = p
    if predictions.ndim == 1 or predictions.shape[1] == 1:
        pred = np.round(predictions)
    else:
        pred = np.argmax(predictions, axis=1)

    accuracy = (pred == labels).mean()
    return {"accuracy": accuracy}
```

**Explanation:**

*   **`compute_metrics(p)`**:  This function calculates the accuracy of the model's predictions.
    *   `predictions, labels = p`:  `p` is a `PredictionOutput` object from the `Trainer`, containing the model's predictions and the true labels.
    *   The code handles both binary and multi-class classification scenarios. For binary classification, it rounds the predictions to 0 or 1. For multi-class, it takes the argmax along the axis 1 to get the predicted class.
    *   `accuracy = (pred == labels).mean()`:  Calculates the mean accuracy by comparing the predicted labels to the true labels.

### 5. Untrained Model Predictions

```python
text_list = ["it was a good movie", "good god,thats the worst film", "better than first film", "not worth the time"]
print("untrained predictions")
print("*" * 50)
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt")
    inputs = inputs.to(device)

    logits = model(inputs).logits
    predictions = torch.argmax(logits)
    print(text + " - " + id2label[predictions.tolist()])
```

**Explanation:**

*   This section demonstrates the model's performance *before* fine-tuning.
*   It iterates through a list of sample sentences.
*   `tokenizer.encode(...)`:  Tokenizes the text and converts it to a PyTorch tensor.
*   `model(inputs).logits`:  Passes the input through the model and retrieves the logits (the raw output of the classification head).
*   `torch.argmax(logits)`:  Finds the class with the highest logit score.
*   `id2label[...]`:  Maps the predicted class ID to its string label (e.g., "Positive" or "Negative").

### 6. LoRA Configuration

```python
peft_config = LoraConfig(
    task_type="SEQ_CLS",
    r=4,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=['q_lin']
)
```

**Explanation:**

*   This section configures the LoRA adapters.
*   `task_type="SEQ_CLS"`: Specifies that this is a sequence classification task.
*   `r=4`:  The rank of the LoRA matrices.  A smaller rank means fewer trainable parameters.  This is the key parameter controlling the size of the LoRA adapters.
*   `lora_alpha=32`: A scaling factor that amplifies the LoRA updates. It's multiplied by the LoRA weights during the forward pass.  Higher values can sometimes lead to better performance, but it requires tuning.
*   `lora_dropout=0.1`:  Dropout rate for the LoRA layers (for regularization).
*   `target_modules=['q_lin']`:  Specifies the modules in the model to which LoRA adapters will be added.  `q_lin` refers to the query projection layer in the self-attention mechanism.  By targeting only the query projection, we're focusing the adaptation on *what* the model attends to.

### 7. Trainable Parameter Count

```python
def print_trainable_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Non-trainable parameters: {total_params - trainable_params}")

print_trainable_parameters(model)
```

**Explanation:**

*   This function calculates and prints the total number of parameters in the model, the number of trainable parameters, and the number of non-trainable parameters.  This is useful for verifying that LoRA is working as expected and that only a small fraction of the parameters are being trained.

### 8. Trainer Initialization and Training

```python
# creater trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,  # this will dynamically pad examples in each batch to be equal length
    compute_metrics=compute_metrics,
)

# train model
trainer.train()
```

**Explanation:**

*   **`Trainer(...)`**:  Creates a `Trainer` object, which handles the training loop.
    *   `model`: The model to train (DistilBERT with LoRA adapters).
    *   `args`: The `TrainingArguments` object, which specifies the training parameters.
    *   `train_dataset`: The training dataset.
    *   `eval_dataset`: The validation dataset.
    *   `tokenizer`: The tokenizer.
    *   `data_collator`:  The `DataCollatorWithPadding` object, which pads the sequences in each batch to the same length.
    *   `compute_metrics`:  The `compute_metrics` function, which calculates the accuracy during training.
*   **`trainer.train()`**:  Starts the training process.  The `Trainer` handles the forward and backward passes, optimization, and logging.

### 9. Trained Model Predictions

```python
model.to(device)
print("Trained model predictions:")
print("--------------------------")
for text in text_list:
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    logits = model(inputs).logits
    predictions = torch.max(logits, 1).indices

    print(text + " - " + id2label[predictions.tolist()[0]])
```

**Explanation:**

*   This section demonstrates the model's performance *after* fine-tuning.
*   It's similar to the "Untrained Model Predictions" section, but it uses the fine-tuned model.
*   The key difference is that the fine-tuned model should now be much better at predicting the sentiment of movie reviews.

###  Summary of Training Results

```
| Epoch | Training Loss | Validation Loss | Accuracy |
|-------|---------------|-----------------|----------|
| 1     | 0.684100      | 0.670383        | 0.663000 |
| 2     | 0.660500      | 0.631024        | 0.837000 |
| 3     | 0.606600      | 0.541647        | 0.868000 |
| 4     | 0.493500      | 0.395706        | 0.879000 |
| 5     | 0.376700      | 0.310003        | 0.878000 |
| 6     | 0.329000      | 0.287510        | 0.878000 |
| 7     | 0.309600      | 0.290163        | 0.876000 |
| 8     | 0.313900      | 0.294083        | 0.879000 |
| 9     | 0.329100      | 0.300639        | 0.876000 |
| 10    | 0.324400      | 0.303185        | 0.875000 |
```

**Analysis:**

*   The table shows the training loss, validation loss, and accuracy at each epoch.
*   The accuracy increases rapidly in the first few epochs and then plateaus.  This suggests that the model is learning quickly and that further training might not lead to significant improvements.  **Early stopping** could be implemented to prevent overfitting.
*   The validation loss decreases initially and then starts to increase slightly, which is another sign of potential overfitting.

### Conclusion

This notebook provides a complete example of how to fine-tune a pre-trained DistilBERT model for sentiment analysis using LoRA.  It demonstrates the key steps involved, from loading the data to evaluating the model's performance.  The use of LoRA allows for efficient fine-tuning with limited computational resources.

### Potential Improvements

*   **Hyperparameter Tuning:** Experiment with different LoRA configurations (rank, alpha, dropout) and training parameters (learning rate, batch size).
*   **Early Stopping:** Implement early stopping to prevent overfitting.
*   **Data Augmentation:**  Increase the size of the training dataset using data augmentation techniques.
*   **Different Base Models:**  Try fine-tuning other pre-trained models (e.g., BERT, RoBERTa).
*   **More Metrics:**  Calculate additional metrics, such as precision, recall, and F1-score.
*   **Error Analysis:**  Analyze the cases where the model makes incorrect predictions to identify areas for improvement.
