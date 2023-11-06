# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

# https://www.kaggle.com/code/madz2000/sentiment-analysis-89-accuracy/notebook
# https://github.com/manthanpatel98/Restaurant-Review-Sentiment-Analysis/blob/master/model.py
# https://www.kaggle.com/code/arunmohan003/sentiment-analysis-using-lstm-pytorch

import numpy as np # linear algebra
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import SequentialFeatureSelector # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer,WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.tokenize import word_tokenize,sent_tokenize
import string
import re
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# File path and name
file_path = 'logs/logfile.txt'
training_file_path = 'logs/trainingData.txt'
testing_file_path = 'logs/testingData.txt'

text_content = ''
# Open the file in append mode
log_file = open(file_path, 'a')
log_file_training = open(training_file_path, 'a')
log_file_testing = open(testing_file_path, 'a')


# Log some initial content
log_file.write("Logging started:\n")


import os
for dirname, _, filenames in os.walk('dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        log_file.write(os.path.join(dirname, filename) + "\n")

# Importing Dataset
df = pd.read_csv('dataset/Musical_instruments_reviews.csv', encoding = "ISO-8859-1")

# Removing the unwanted columns from the dataframe and combining reviewText and summary column to one column and naming it as text
df = df.drop(columns=["reviewerID","asin","reviewerName","helpful","unixReviewTime", "reviewTime"])
df['text'] = df['reviewText'] + ' ' + df['summary']
del df['reviewText']
del df['summary']

print(df.head)
print(df.overall.value_counts())
log_file.write(f"{df.head} \n")
log_file.write(f"{df.overall.value_counts()}\n")



# 5.0    6938
# 4.0    2084
# 3.0     772
# 2.0     250
# 1.0     217
# Name: overall, dtype: int64

def sentiment_rating(rating):
    # Replacing ratings of 1,2,3 with 0 (not good) and 4,5 with 1 (good)
    if(int(rating) == 1 or int(rating) == 2 or int(rating) == 3):
        return 0
    else: 
        return 1
    
df.overall = df.overall.apply(sentiment_rating) 

print(df.head)
log_file.write(f"{df.head}\n")

# 0            1  Not much to write about here, but it does exac...
# 1            1  The product does exactly as it should and is q...
# 2            1  The primary job of this device is to block the...
# 3            1  Nice windscreen protects my MXL mic and preven...

print(df.overall.value_counts())
log_file.write(f"{df.overall.value_counts() }\n")
# 1    9022
# 0    1239
# Name: overall, dtype: int64

data = df  #  dataset
labels = ['overall','text']

class BuildDataSet(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Implement logic to return a sample
        # Return a tuple (sample, target) or a dictionary {'data': sample, 'target': target}
        return (self.data['overall'][idx], self.data['text'][idx])


#Finding stop words
stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)

def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

# Lemmatizing the words
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
    final_text = []
    if isinstance(text, str):  # Check if text is a string
        for i in text.split():
            
            if i.strip().lower() not in stop:
                pos = pos_tag([i.strip()])
                word = lemmatizer.lemmatize(i.strip(),get_simple_pos(pos[0][1]))
                final_text.append(word.lower())
        return " ".join(final_text)
    else:
     return " " 
    
    
df.text = df.text.apply(lemmatize_words)
print('-------------- Affter lemmatization ------------------')
print(df.head)

log_file.write('-------------- Affter lemmatization ------------------' + "\n")
log_file.write(f"{df.head }\n")

# -------------- Affter lemmatization ------------------
# <bound method NDFrame.head of        overall                                               text
# 0            1  much write here, exactly suppose to. filter po...
# 1            1  product exactly quite affordable.i realize dou...
# 2            1  primary job device block breath would otherwis...
# 3            1  nice windscreen protects mxl mic prevents pops...
# 4            1  pop filter great. look performs like studio fi...
# ...        ...                                                ...
# 10256        1             great, expected. thank all. five stars
# 10257        1  i've think try nanoweb string while, bit put h...
# 10258        1  try coat string past include elixirs) never fo...
# 10259        1  well, made elixir developed taylor guitars ......
# 10260        1  string really quite good, call perfect. unwoun...

# [10261 rows x 2 columns]>

# Split the dataset into training and testing indices
# x_train, x_test, y_train, y_test = train_test_split(list(range(len(dataset_to_use))), test_size=0.2, random_state=42)
x_train,x_test,y_train,y_test = train_test_split(df.text, df.overall,test_size = 0.2 , random_state = 0)

def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s

def tockenize(x_train,y_train,x_val,y_val):
    word_list = []

    stop_words = set(stopwords.words('english')) 
    for sent in x_train:
        for word in sent.lower().split():
            word = preprocess_string(word)
            if word not in stop_words and word != '':
                word_list.append(word)
  
    corpus = Counter(word_list)

    # sorting on the basis of most common words
    corpus_ = sorted(corpus,key=corpus.get,reverse=True)[:1000]

    # creating a dict
    onehot_dict = {w:i+1 for i,w in enumerate(corpus_)}
    
    # tockenize
    final_list_train,final_list_test = [],[]
    for sent in x_train:
            final_list_train.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                     if preprocess_string(word) in onehot_dict.keys()])
    for sent in x_val:
            final_list_test.append([onehot_dict[preprocess_string(word)] for word in sent.lower().split() 
                                    if preprocess_string(word) in onehot_dict.keys()])
            
    encoded_train = [1 if label == 1 else 0 for label in y_train]  
    encoded_test = [1 if label == 0 else 0 for label in y_val] 
    #
    #
    #
    # Assuming final_list_train, final_list_test are lists of sequences of varying lengths
    # Pad sequences to make them uniform in length
    # max_length = max(len(seq) for seq in final_list_train + final_list_test)
    # final_list_train = [seq + [0] * (max_length - len(seq)) for seq in final_list_train]
    # final_list_test = [seq + [0] * (max_length - len(seq)) for seq in final_list_test]

    print('\n-------------final_list_train-------------------\n')
    print(final_list_train)

    log_file_training.write('\n-------------final_list_train start-------------------\n')    
    # Convert each inner list to a string
    list_strings_final_list_train = [str(sublist) for sublist in final_list_train]
    # Merge the list representations into a single string
    string_final_list_train = '\n'.join(list_strings_final_list_train)
    log_file_training.write(string_final_list_train)
    log_file_training.write('\n-------------final_list_train start end-------------------\n')    


    log_file_testing.write('\n-------------final_list_test start-------------------\n')
    list_strings_final_list_test = [str(sublist) for sublist in final_list_test]
    string_final_list_test = '\n'.join(list_strings_final_list_test)
    log_file_testing.write(string_final_list_test)
    log_file_testing.write('\n-------------final_list_test start end-------------------\n')


    log_file.write('\n-------------encoded_test start -------------------\n')
    list_strings_encoded_test = [str(sublist) for sublist in encoded_test]
    string_final_encoded_test = ' , '.join(list_strings_encoded_test)
    log_file.write(string_final_encoded_test)
    log_file.write('\n-------------encoded_test end-------------------\n')


    log_file.write('\n-------------onehot_dict start-------------------\n')
    list_strings_onehot_dict = [str(sublist) for sublist in onehot_dict]
    string_final_onehot_dict = ' , '.join(list_strings_onehot_dict)
    log_file.write(string_final_onehot_dict)
    log_file.write('\n-------------onehot_dict end-------------------\n')



    # Convert to NumPy arrays
    np_final_list_train = np.array(final_list_train)
    np_final_list_test = np.array(final_list_test)

    return np_final_list_train, np.array(encoded_train), np_final_list_test, np.array(encoded_test), onehot_dict

good = x_train[y_train[y_train == 1].index]
bad = x_train[y_train[y_train == 0].index]

x_train.shape, good.shape, bad.shape
print('-------------shapes-------------------')
print(x_train.shape, good.shape, bad.shape)
# ((8208,), (7197,), (1011,))
print('-------------good-------------------')
print(good[0])
print('-------------bad-------------------')
print(bad)

print('-------------- train_indices ------------------')
print(len(y_train)) 
# 8208

print('-------------- test_indices ------------------')
print(len(y_test)) 
# 2053

log_file.write('-------------shapes-------------------\n')
log_file.write(f"{x_train.shape, good.shape, bad.shape}\n")
# ((8208,), (7197,), (1011,))
log_file.write('-------------good-------------------\n')
log_file.write(f"{good[0]}\n")
log_file.write('-------------bad-------------------\n')
log_file.write(f"{bad}\n")

log_file.write('-------------- train_indices ------------------\n')
log_file.write(f"{len(y_train)}\n") 
# 8208

log_file.write('-------------- test_indices ------------------\n')
log_file.write(f"{len(y_test)}\n") 
# 2053



# Text Reviews with Poor Ratings
plt.figure(figsize = (20,20)) 
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(bad))
plt.imshow(wc,interpolation = 'bilinear')
plt.title('Text Reviews with Bad Ratings')
plt.savefig('Text_Bad_Ratings.png')

# Text Reviews with Good Ratings
plt.figure(figsize = (20,20)) 
wc = WordCloud(min_font_size = 3,  max_words = 3000 , width = 1600 , height = 800).generate(" ".join(good))
plt.imshow(wc,interpolation = 'bilinear')
plt.title('Text Reviews with Good Ratings')
plt.savefig('Text_Good_Ratings.png')

plt.figure(figsize = (20,20)) 
dd = pd.Series(y_train).value_counts()
print(f'dd is : {dd}')

sns.barplot(x=np.array(['positive','negative']),y=dd.values)
plt.savefig('Good Ratings Vs Bad Ratings.png')

# Tokenize after representation
x_train, y_train, x_test, y_test, vocab = tockenize(x_train, y_train, x_test, y_test)

print(f'Length of vocabulary is {len(vocab)}')
log_file.write(f'\n Length of vocabulary is : {len(vocab)} \n')

# Length of vocabulary is 1000

# Analysing review length
rev_len = [len(i) for i in x_train]
print(f'rev_len is : {rev_len}')
plt.figure(figsize = (20,20)) 
pd.Series(rev_len).hist()
plt.savefig('ReviewLength.png')
print(f'Description of the data : {pd.Series(rev_len).describe()}')
log_file.write(f' \n \n Description of the data : {pd.Series(rev_len).describe()} \n')

# Description of the data : 
# count    8208.000000
# mean       36.255970
# std        40.175509
# min         0.000000
# 25%        15.000000
# 50%        24.000000
# 75%        42.000000
# max       745.000000
# dtype: float64

# Observations :
# a) Mean review length = around 36.
# b) minimum length of reviews is 0.
# c)There are quite a few reviews that are extremely long, we can manually investigate them to check whether we need to include or exclude them from our analysis.

# padding each of the sequence to max length
def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

# we have very less number of reviews with length > 500. So we will consider only those below it.
x_train_pad = padding_(x_train,500)
x_test_pad = padding_(x_test,500)

# creating Tensor datasets
train_data = TensorDataset(torch.from_numpy(x_train_pad), torch.from_numpy(y_train))
valid_data = TensorDataset(torch.from_numpy(x_test_pad), torch.from_numpy(y_test))

# dataloaders
batch_size = 50

# making sure to SHUFFLE the data
train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

# obtaining one batch of training data
dataiter = iter(train_loader)
sample_x, sample_y = next(dataiter)

print('Sample input size: ', sample_x.size()) # batch_size, seq_length
print('Sample input: \n', sample_x)
print('Sample input: \n', sample_y)

log_file.write(f' \n Sample input size: : {sample_x.size()} \n')
log_file.write(f' \n Sample input size: sample_x : {sample_x} \n')
log_file.write(f' \n Sample input size: sample_y : {sample_y} \n')

# Sample input size:  torch.Size([50, 500])
# Sample input: 
#  tensor([[  0,   0,   0,  ..., 106, 600, 183],
#         [  0,   0,   0,  ...,  90, 661,   7],
#         [  0,   0,   0,  ..., 172,  79, 367],
#         ...,
#         [  0,   0,   0,  ...,  44,   3,  38],
#         [  0,   0,   0,  ...,  48,  14, 262],
#         [  0,   0,   0,  ..., 894,   3,  28]], dtype=torch.int32)
# Sample input:
#  tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1,
#         0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1,
#         1, 1], dtype=torch.int32)

# Model
no_layers = 2
vocab_size = len(vocab) + 1 #extra 1 for padding
embedding_dim = 64
output_dim = 1
hidden_dim = 256

class SentimentRNN(nn.Module):
    def __init__(self, no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5):
        super(SentimentRNN,self).__init__()
 
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
 
        self.no_layers = no_layers
        self.vocab_size = vocab_size
    
        # embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        #lstm
        self.lstm = nn.LSTM( input_size = embedding_dim,
                             hidden_size = self.hidden_dim,
                             num_layers = no_layers,
                             batch_first = True)
        
        
        # dropout layer
        self.dropout = nn.Dropout(0.3)
    
        # linear and sigmoid layer
        self.fc = nn.Linear(self.hidden_dim, output_dim)
        self.sig = nn.Sigmoid()
        
    def forward(self,x,hidden):
        batch_size = x.size(0)

        # embeddings and lstm_out
        embeds = self.embedding(x)
        
        # shape: B x S x Feature   since batch = True
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim) 
        
        # dropout and fully connected layer
        out = self.dropout(lstm_out)
        out = self.fc(out)
        
        # sigmoid function
        sig_out = self.sig(out)
        
        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)

        sig_out = sig_out[:, -1] # get last batch of labels
        
        # return last sigmoid output and hidden state
        return sig_out, hidden
        
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        h0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        c0 = torch.zeros((self.no_layers,batch_size,self.hidden_dim)).to(device)
        hidden = (h0,c0)
        return hidden

              
              
# If we have a GPU available, we'll set our device to GPU. 

# # compose the LSTM Network

torch.manual_seed(1)
is_cuda = torch.cuda.is_available()

if is_cuda:
    device = torch.device("cuda")
    print("GPU is available")
    log_file.write(f' GPU is available \n')

else:
    device = torch.device("cpu")
    print("GPU not available, CPU used")
    log_file.write(f'GPU not available, CPU used \n')



model = SentimentRNN(no_layers, vocab_size, hidden_dim, embedding_dim, drop_prob=0.5)

#moving to gpu
model.to(device)

print(model)
log_file.write(f' \n model: : {model} \n')

# Close the file when done logging
log_file.close()

print(f"Logging completed. File '{file_path}' has been appended with new content.")
################################################################
# vectorizer = CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
# vectorizer.fit(df.text)

# X_train_vec = vectorizer.transform(x_train)
# X_test_vec = vectorizer.transform(x_test)

# print('-------------X_train_vec-------------------')
# # print(X_train_vec)

# print('-------------X_test_vec-------------------')
# # print(X_test_vec)

# print('-------------X_train_vec.shape-------------')
# print(X_train_vec.shape)

# print('-------------X_test_vec.shape-------------')
# print(X_test_vec.shape)




# # If we have a GPU available, we'll set our device to GPU. We'll use this device variable later in our code.
# if is_cuda:
#     device = torch.device("cuda")
#     print("GPU is available")
# else:
#     device = torch.device("cpu")
#     print("GPU not available, CPU used")

# # create Tensor datasets

# # Assuming X_train_vec is  csr_matrix and y_train is a numpy array
# X_train_array = X_train_vec.toarray()
# X_test_array = X_test_vec.toarray()


# # Create the TensorDataset
# train_data = TensorDataset(torch.from_numpy(X_train_array), torch.from_numpy(y_train))
# valid_data = TensorDataset(torch.from_numpy(X_test_array), torch.from_numpy(y_test))

# print('-------------train_data-------------')
# print(train_data)
# print('-------------valid_data-------------')
# print(valid_data)



# # dataloaders
# batch_size = 50

# # make sure to SHUFFLE your data
# train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
# valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)

# # obtain one batch of training data
# dataiter = iter(train_loader)
# sample_x, sample_y = dataiter.next()

# print('Sample input size: ', sample_x.size()) # batch_size, seq_length
# print('Sample input: \n', sample_x)
# print('Sample input: \n', sample_y)

