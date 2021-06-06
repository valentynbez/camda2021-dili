import argparse
# Standard DS & plotting libraries
import numpy as np
import pandas as pd
# Serialization
import pickle

import random
import time
# Deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel,  AutoTokenizer
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
# ML utils 
from sklearn.model_selection import train_test_split


def set_seed(seed_value=42):
    
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    
    
def preprocessing_for_bert(tokenizer, strings):
   
    input_ids = []
    attention_masks = []

    for article in strings:
        encoded_article = tokenizer.encode_plus(
            text=article,                   
            add_special_tokens=True,        
            max_length=512,                 
            truncation=True,
            padding="max_length",           
            return_attention_mask=True     
            )
        
        input_ids.append(encoded_article.get('input_ids'))
        attention_masks.append(encoded_article.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks   


class BertClassifier(nn.Module):

    def __init__(self, model_path, freeze_bert=False):

        super(BertClassifier, self).__init__()
        # Specify hidden size of pretrained model, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        self.bert = AutoModel.from_pretrained(model_path)

        # One layer linear unit for fine-tuning
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Linear(H, D_out)
        )

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
    def forward(self, input_ids, attention_mask):
        
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        last_hidden_state = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state)

        return logits 

    
def initialize_model(epochs=4):

    bert_classifier = BertClassifier(model_path, freeze_bert=False)

    # Move tensors to GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=5e-5,   
                      eps=1e-8   
                      )

    total_steps = len(train_dataloader) * epochs

    # Set up the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    
    return bert_classifier, optimizer, scheduler





def train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):

    print("Start training...\n")
    for epoch_i in range(epochs):

        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*70)

        # Measure the time
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0

        # Put the model into the training mode
        model.train()

        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
             
                time_elapsed = time.time() - t0_batch

                # Print results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        avg_train_loss = total_loss / len(train_dataloader)

        print("-"*70)

        if evaluation == True:
            # After each epoch measure performance on validation dataset
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f}")
            print("-"*70)
        print("\n")
    
    print("Training complete!")


def evaluate(model, val_dataloader):

    # Disable dropout layers
    model.eval()
    
    val_accuracy = []
    val_loss = []

    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Calculate loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate accuracy on training set
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Accuracy and loss on validation
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


def bert_predict(model, test_dataloader, device):

    model.eval()

    all_logits = []


    for batch in test_dataloader:

        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    all_logits = torch.cat(all_logits, dim=0)

    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    return probs


if __name__ == "__main__": 
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION]",
        
        description="Train any of the 5 pretrained biomedical BERT models from Hugging Face library"
    )
    
    args = parser.parse_args()
    
    models = ['allenai/scibert_scivocab_uncased', 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext',
              'emilyalsentzer/Bio_ClinicalBERT', 'dmis-lab/biobert-v1.1', 'allenai/biomed_roberta_base', ]

    if torch.cuda.is_available():       
        device = torch.device("cuda")

    for model_path in models:
        
        # For fine-tuning BERT, the authors recommend a batch size of 16-32, but our RTX could hold only 8. 
        batch_size = 8

        positive = pd.read_csv('data/positive.tsv', sep='\t', index_col=0)
        positive['target'] = 1
        negative = pd.read_csv('data/negative.tsv', sep='\t', index_col=0)
        negative['target'] = 0
        data = positive.append(negative)
        data['concat'] = data.Title.map(str) + " " + data.Abstract.fillna(' ').map(str)
        data['bert'] = data['concat'].apply(lambda x: x.lower())

        tokenizer = AutoTokenizer.from_pretrained(model_path, do_lower_case=True)
        print('Tokenizing data...')
        
        train_inputs, train_masks = preprocessing_for_bert(tokenizer, data.bert)
        train_labels = torch.tensor(data.target.values)

        # Create the DataLoader for our training set
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size, num_workers=1)

        # Loss function
        loss_fn = nn.CrossEntropyLoss()
        
        set_seed(42)
        bert_classifier, optimizer, scheduler = initialize_model(epochs=4)
        train(bert_classifier, train_dataloader, epochs=4)
        model_name = 'full-' + '-'.join(model_path.split('/'))

        # Save the model
        torch.save(bert_classifier, f"./models/{model_name}.pkl", pickle_module=pickle)

