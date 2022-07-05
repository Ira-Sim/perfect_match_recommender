import os
import matplotlib.pyplot as plt
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
from transformers import BertTokenizer, BertModel
DATA_DIR = '/home/mila/c/chris.emezue/scratch/hackbay'

device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
 
LR = 0.0001
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

INDEX_TO_AGE = {v:k for k,v in AGE_TO_INDEX.items()}
INDEX_TO_GENDER = {v:k for k,v in GENDER_TO_INDEX.items()}

GENDER_CLASSES = len(GENDER_TO_INDEX)
AGE_CLASSES = len(AGE_TO_INDEX)
BATCH_SIZE=128

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
  text = re.sub(r'[\n\t\r]',' ',text)
  
  # trim to required length
  text = text[:MAX_SEQ_LEN]
  return text

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
    
    state_dict = torch.load(load_path,map_location=torch.device('cpu'))
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

"""# Setup model..."""


def load_model():
    model = PerfectMatch().to(device)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    metrics_checkpoint = os.path.join(DATA_DIR,'perfect_match_model.pt')  
    if os.path.exists(metrics_checkpoint):
        model,optimizer = load_checkpoint(metrics_checkpoint, model, optimizer)

    model = model.to(device)
    return model



def predict(text,model,tokenizer,distribution=True): #Given text, it predicts the age and gender distribution.
  softmax = nn.Softmax(dim=1)
  
  text = preprocess_text(text)
  model.eval()
  with torch.no_grad():
    sent1_ = tokenizer.encode(text)
   
    sent1_ = torch.LongTensor(sent1_).unsqueeze(0).to(device)
    output_age, output_gender = model.forward(sent1_)
    preds_age=softmax(output_age)
    preds_gender = softmax(output_gender)

    if not distribution:

        rel_preds_age = torch.argmax(preds_age,dim=1).squeeze().cpu()
        rel_preds_gender = torch.argmax(preds_gender,dim=1).squeeze().cpu()

        rel_age = list(AGE_TO_INDEX.keys())[list(AGE_TO_INDEX.values()).index(rel_preds_age)] 
        rel_gender = list(GENDER_TO_INDEX.keys())[list(GENDER_TO_INDEX.values()).index(rel_preds_gender)]
        return rel_age, rel_gender
    else:
        # Return a distribution 
        age_pred_distribution = preds_age.squeeze().cpu().numpy().tolist()    
        age_pred_labels = [INDEX_TO_AGE[i] for i in range(len(age_pred_distribution))]

        gender_pred_distribution = preds_gender.squeeze().cpu().numpy().tolist()    
        gender_pred_labels = [INDEX_TO_GENDER[i] for i in range(len(gender_pred_distribution))]
        return {'age':{'labels':age_pred_labels,'probs':age_pred_distribution},
                'gender': {'labels':gender_pred_labels,'probs':gender_pred_distribution}}

