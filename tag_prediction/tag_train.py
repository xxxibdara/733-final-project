import random

import numpy as np
import torch
from sklearn.preprocessing import MultiLabelBinarizer
from torch import nn
from torch.nn import (LSTM, RNN, BCELoss, Dropout, Embedding, Linear, ReLU,
                      Sequential, Sigmoid)
from torchtext import data
from torchtext.data import Field, TabularDataset

import tag_preprocessing as tp


# define function to fetch text from data
def fetch_text(examples):
    text = []
    for example in examples:
        query = vars(example)['query']
        text.append(query)
    return text

# padding and converting to numers
def convert2seq(text):
    text = TEXT.pad(text)
    text = TEXT.numericalize(text)
    return text

# define function to fetch tags from data
def fetch_tags(data):
    tags = []
    for example in data.examples:
        tags.append(vars(example)['tags'])
    return tags


# define hyperparameters
max_len = 100
TEXT = data.Field(tokenize=tp.cleaner, batch_first=True, fix_length=max_len)
LABEL = data.LabelField(batch_first=True)
fields = [('query', TEXT), ('tags', LABEL)]

# load data
training_data = TabularDataset(
    path='tabular_data.csv', format='csv', fields=fields, skip_header=True)

# split data into train and validation
train_data, valid_data = training_data.split(
    split_ratio=0.8, random_state=random.seed(32))

# build vocabulary
TEXT.build_vocab(train_data, min_freq=3)

# fetch text from data
train_text = fetch_text(train_data)
valid_text = fetch_text(valid_data)

# padding and converting to numers
X_train = convert2seq(train_text)
X_valid = convert2seq(valid_text)

# fetch tags from data
train_tags = fetch_tags(train_data)
valid_tags = fetch_tags(valid_data)

# convert tags to one-hot encoding
train_tags_list = [i.split(",") for i in train_tags]
valid_tags_list = [i.split(",") for i in valid_tags]

# define MultiLabelBinarizer
mlb = MultiLabelBinarizer()
mlb.fit(train_tags_list)

# fit and transform tags
y_train = mlb.transform(train_tags_list)
y_valid = mlb.transform(valid_tags_list)
y_train = torch.FloatTensor(y_train)
y_valid = torch.FloatTensor(y_valid)

# define model
emb = Embedding(num_embeddings=len(TEXT.vocab), embedding_dim=50)
sample_embedding = emb(X_train[:1])
rnn = RNN(input_size=50, hidden_size=128,
          batch_first=True, nonlinearity='relu')
hidden_states, last_hidden_state = rnn(sample_embedding)
reshaped = hidden_states.reshape(hidden_states.size(0), -1)


class Net(nn.Module):
    # define all the layers used in model
    def __init__(self):
        # Constructor
        super(Net, self).__init__()
        self.rnn_layer = nn.Sequential(
            Embedding(num_embeddings=len(TEXT.vocab), embedding_dim=50),
            RNN(input_size=50, hidden_size=128,
                nonlinearity='relu', batch_first=True)
        )

        self.dense_layer = nn.Sequential(
            Linear(12800, 128),
            ReLU(),
            Linear(128, 10),
            Sigmoid()
        )

    def forward(self, x):
        # define the forward pass
        hidden_states, last_hidden_state = self.rnn_layer(x)
        hidden_states = hidden_states.reshape(hidden_states.size(0), -1)
        outputs = self.dense_layer(hidden_states)
        return outputs


model = Net()
optimizer = torch.optim.Adam(model.parameters())
criterion = BCELoss()

# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

# define training function


def train(X, y, batch_size):
    model.train()
    epoch_loss = 0
    no_of_batches = 0
    indices = torch.randperm(len(X))

    for i in range(0, len(indices), batch_size):
        ind = indices[i:i+batch_size]
        batch_x = X[ind]
        batch_y = y[ind]

        if torch.cuda.is_available():
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        optimizer.zero_grad()

        outputs = model(batch_x)
        outputs = outputs.squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss = epoch_loss + loss.item()
        no_of_batches = no_of_batches+1

    return epoch_loss/no_of_batches

# define evaluation function


def evaluate(X, y, batch_size):
    model.eval()
    epoch_loss = 0
    no_of_batches = 0
    indices = torch.randperm(len(X))

    with torch.no_grad():
        for i in range(0, len(indices), batch_size):
            ind = indices[i:i+batch_size]

            batch_x = X[ind]
            batch_y = y[ind]
            if torch.cuda.is_available():
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()

            outputs = model(batch_x)
            outputs = outputs.squeeze()

            loss = criterion(outputs, batch_y)
            epoch_loss = epoch_loss + loss.item()
            no_of_batches = no_of_batches + 1

        return epoch_loss/no_of_batches

# define prediction function


def predict(X, batch_size):
    model.eval()
    predictions = []
    indices = torch.arange(len(X))

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            ind = indices[i:i+batch_size]
            batch_x = X[ind]

            if torch.cuda.is_available():
                batch_x = batch_x.cuda()

            outputs = model(batch_x)
            outputs = outputs.squeeze()
            prediction = outputs.data.cpu().numpy()
            predictions.append(prediction)

    predictions = np.concatenate(predictions, axis=0)

    return predictions


# define training parameters
N_EPOCHS = 10
batch_size = 32

# intialization
best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):
    train_loss = train(X_train, y_train, batch_size)
    valid_loss = evaluate(X_valid, y_valid, batch_size)

    print('\nEpoch :', epoch,
          'Training loss:', round(train_loss, 4),
          '\tValidation loss:', round(valid_loss, 4))
    # save the best training model
    if best_valid_loss >= valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), './tag_model')
        print("\n-------------Saved best model--------------")
