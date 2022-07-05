import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torchtext
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
import re
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,f1_score
import pandas as pd

DATA_DIR =r'C:\Users\vi04wecu\Desktop\Hackbay\processed_data'

# Ira: To be changed?
device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')

#Parameters for training
LR = 0.0001             # too small CHANGE TO 0.001

from transformers import AutoTokenizer, AutoModel
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
 

GENDER_TO_INDEX = {
    'maennlich':0,
    'weiblich':1
}

AGE_TO_INDEX = {
'16 bis 17 Jahre':0,
 '50 bis 54 Jahre':1,
 '65 bis 69 Jahre':2, 
 '25 bis 29 Jahre':3,
 '14 bis 15 Jahre':4,
 '55 bis 59 Jahre':5,
 '10 bis 13 Jahre':6,
 '75 und mehr Jahre':7,
 '60 bis 64 Jahre':8,
 '35 bis 39 Jahre':9,
 '40 bis 44 Jahre':10,
 '70 bis 74 Jahre':11,
 '30 bis 34 Jahre':12,
 '45 bis 49 Jahre':13,
 '18 bis 19 Jahre':14,
 '20 bis 24 Jahre':15
 }


GENDER_CLASSES = len(GENDER_TO_INDEX)
AGE_CLASSES = len(AGE_TO_INDEX)
BATCH_SIZE=1024

TEXT_COL_NAME='text'
GENDER_COL_NAME='gender'
AGE_COL_NAME='age'
MAX_SEQ_LEN=512


#tokenizer = AutoTokenizer.from_pretrained(model_type)

def preprocess_text(text):
  # preprocess text.
  # remove non-alphanumeric characters
  # keep numbers
  text = re.sub(r'\W+',' ',text,flags=re.UNICODE)
  text = re.sub(r'[\n\t\r]',' ',text)            # delete linebreakers on windows, linux, mac?
  
  # trim to required length
  text = text[:MAX_SEQ_LEN]
  return text

if not os.path.exists('train.csv'):
    train_df = pd.read_excel('hackbay_train_dataset.xlsx').fillna('')

    train_df['age'] = train_df['age'].apply(lambda x:AGE_TO_INDEX[x])
    train_df['gender'] = train_df['gender'].apply(lambda x:GENDER_TO_INDEX[x])
    train_df['text'] =  train_df['title']+ ' '+ train_df['text']
    train_df['text'] = train_df['text'].apply(lambda x: preprocess_text(' '.join(x.split('|'))))

    tr_df = train_df.drop(['url_id', 'title','keywords','colors','number_of_images','hashed_id'], axis = 1)

    valid_size = 0.2
    valid_len = int(valid_size*tr_df.shape[0])
    valid_indices = [i for i in range(valid_len)]
    train_indices = [i for i in range(len(tr_df)) if i not in valid_indices]

    train = tr_df.iloc[train_indices]
    train.to_csv('train.csv',index=False)

    valid = tr_df.iloc[valid_indices]
    valid.to_csv('valid.csv',index=False)

def encode_text(text):
  return tokenizer.encode(text = text,return_tensors = 'pt')

# Save and Load Functions

def save_checkpoint(save_path, model, optimizer, valid_loss):

    if save_path == None:
        return
    
    state_dict = {'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'valid_loss': valid_loss}
    
    torch.save(state_dict, save_path)


def load_checkpoint(load_path, model, optimizer):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')
    
    model.load_state_dict(state_dict['model_state_dict'])
    optimizer.load_state_dict(state_dict['optimizer_state_dict'])
    
    return model,optimizer


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):

    if save_path == None:
        return
    
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}
    
    torch.save(state_dict, save_path)
    #print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):

    if load_path==None:
        return
    
    state_dict = torch.load(load_path)
    print(f'Model loaded from <== {load_path}')
    
    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']


def compute_metrics(labels,probs):
  softmax = nn.Softmax(dim=1)
  preds =softmax(probs)
  acc_preds = torch.argmax(preds,dim=1).squeeze().cpu().tolist()
  labels = labels.squeeze().cpu().tolist()
  acc = accuracy_score(labels,acc_preds)
  f1 = f1_score(labels,acc_preds,average='weighted')
  return {'f1': f1, 'accuracy':acc}

"""# Dataset Preparation for Model"""

# Model parameter

PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

# Fields

#For input text
input_text = Field(lower=False,use_vocab=False,include_lengths=False, batch_first=True,tokenize=tokenizer.encode,pad_token=PAD_INDEX, unk_token=UNK_INDEX)


#For the age and gender labels
age_label = Field(sequential=False, use_vocab=False, batch_first=True,is_target=True)
gender_label = Field(sequential=False, use_vocab=False, batch_first=True,is_target=True)



fields = [(AGE_COL_NAME, age_label),(GENDER_COL_NAME, gender_label),(TEXT_COL_NAME, input_text)]

# TabularDataset

train, valid= TabularDataset.splits(path=DATA_DIR, train='train.csv', validation='valid.csv',format='CSV', fields=fields, skip_header=True)

# Iterators

train_iter = BucketIterator(train, batch_size=BATCH_SIZE,device=device, train=True)
valid_iter = BucketIterator(valid, batch_size=BATCH_SIZE,device=device, train=True)


#test_iter = Iterator(test, batch_size=BATCH_SIZE,device=device, train=False,sort=False,shuffle=False,sort_within_batch=False)

"""# Model Configuration: PerfectMatch Classifier"""

class PerfectMatch(nn.Module): # Jointly predicts the age and gender. 

    def __init__(self, dimension=128,num_layers=1,dropout=0.1):
        super(PerfectMatch, self).__init__()
        

        #self.BERT_Embedding_model = AutoModel.from_pretrained(model_type)
        self.BERT_Embedding_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
        self.dimension = dimension
        self.lstm = nn.LSTM(input_size=1,    #Because we are concatenating two BioBERT embeddings
                            hidden_size=self.dimension,
                            num_layers=num_layers,
                            bidirectional=True)
        
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(1536*self.dimension,self.dimension)
      
        self.fc_age = nn.Linear(self.dimension, AGE_CLASSES)
        self.fc_gender = nn.Linear(self.dimension, GENDER_CLASSES)
        #self.softmax = torch.nn.Softmax() # don't need this if we are using nn.CrossEntropyLoss()

        # Freeze BioBert which is used as embedding model
        for param in self.BERT_Embedding_model.parameters():
          param.requires_grad = False

    def forward(self, text):

        x = self.BERT_Embedding_model.forward(input_ids=text).pooler_output #768 dimensions
     
        final_emb=x.unsqueeze(1).transpose(2,1)
        #print(f'Embedding size after transpose: {final_emb.size()}')
       
      
        output, (h_n, c_n) = self.lstm(final_emb)
        #print(f'Output shape: {output.size()}')

        flattened = output.view(output.size(0),-1)
        #print(f"Size of flattened: {flattened.shape} ")
        text_fea = self.fc(flattened)
        #print(f"Size of text fea: {text_fea.shape} ")
        text_fea=self.dropout(text_fea)
        
        x_age = torch.squeeze(self.fc_age(text_fea),1)
        x_gender = torch.squeeze(self.fc_gender(text_fea) ,1)
               
        return x_age,x_gender

def test_eval(test_loader):
  acc_age=0
  f1_age=0
  acc_gender = 0
  f1_gender = 0 
  model.eval()
  with torch.no_grad():                    
    # testing loop
    count=0
    for data in test_loader:
        labels_age = data.age
        labels_gender= data.gender
        text = data.text

        output_age,output_gender = model(text)

        c_age = compute_metrics(labels_age,output_age)
        c_gender = compute_metrics(labels_gender,output_gender)
        count+=1
        acc_age+=c_age['accuracy']
        f1_age+=c_age['f1']
        acc_gender+=c_gender['accuracy']
        f1_gender+=c_gender['f1']
    return 'ACC Age: {:.3f}, F1 Age: {:.3f} | ACC Gender: {:.3f}, F1 Gender: {:.3f} '.format(acc_age/count,f1_age/count,acc_gender/count,f1_gender/count)

"""# Training starts here..."""

# Training Function

def train(model,
          optimizer,
          train_loader = train_iter,
          valid_loader = valid_iter,
          num_epochs = 20,
          eval_every = 10,
          file_path = DATA_DIR,
          best_valid_loss = float("Inf")):
    
    
    criterion_age =nn.CrossEntropyLoss() 
    criterion_gender =nn.CrossEntropyLoss() 
  
    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        for data in train_loader: 
            
            labels_age = data.age.to(device)
            labels_gender= data.gender.to(device)
            text = data.text.to(device)
        
            output_age,output_gender = model(text)

            loss_age = criterion_age(output_age, labels_age)
            loss_gender = criterion_gender(output_gender, labels_gender)
            optimizer.zero_grad()
            loss = loss_age + loss_gender # add both losses equally
            loss.backward()
            optimizer.step()

            # update running values
            running_loss += loss.item()
            global_step += 1

            # evaluation step
            if global_step % eval_every == 0:
                model.eval()
                with torch.no_grad():                    
                  # validation loop
                  for data in valid_loader:
                      labels_age = data.age.to(device)
                      labels_gender= data.gender.to(device)
                      text = data.text.to(device)
                
                      output_age,output_gender = model(text)

                      loss_age_valid = criterion_age(output_age, labels_age)
                      loss_gender_valid = criterion_gender(output_gender, labels_gender)
                      loss = loss_age_valid + loss_gender_valid  
                    
                      valid_running_loss += loss.item()

                      metrics_age= compute_metrics(labels_age,output_age)
                      metrics_gender =  compute_metrics(labels_gender,output_gender)

                # evaluation
                average_train_loss = running_loss / eval_every
                average_valid_loss = valid_running_loss / len(valid_loader)
                train_loss_list.append(average_train_loss)
                valid_loss_list.append(average_valid_loss)
                global_steps_list.append(global_step)

                # resetting running values
                running_loss = 0.0                
                valid_running_loss = 0.0
                model.train()

                # print progress
                print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f} | Metrics Age: {} |  Metrics Gender: {}'
                      .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                              average_train_loss, average_valid_loss,metrics_age,metrics_gender))
                
                # checkpoint
                if best_valid_loss > average_valid_loss:
                    print(f'Improvement from {best_valid_loss} to {average_valid_loss}. Saving model checkpoints')
                    best_valid_loss = average_valid_loss
                    
                    save_checkpoint('perfect_match_model.pt', model, optimizer, best_valid_loss)
                    save_metrics('perfect_match_metrics.pt', train_loss_list, valid_loss_list, global_steps_list)
    
    
    print('Finished Training!')

"""# Setup model..."""

model = PerfectMatch().to(device)

optimizer = optim.Adam(model.parameters(), lr=LR)                   #Ira 
metrics_checkpoint = os.path.join(DATA_DIR,'perfect_match_model.pt')  
if  os.path.exists(metrics_checkpoint):
  model,optimizer = load_checkpoint(metrics_checkpoint, model, optimizer)

model = model.to(device)

"""# Train..."""

train(model=model, optimizer=optimizer, num_epochs=2000)


"""# Plot losses..."""

train_loss_list, valid_loss_list, global_steps_list = load_metrics(os.path.join(DATA_DIR,'perfect_match_metrics.pt'))
plt.plot(global_steps_list, train_loss_list, label='Train')
plt.plot(global_steps_list, valid_loss_list, label='Valid')
plt.xlabel('Global Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_plot.png')
plt.show()
