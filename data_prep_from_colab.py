import pandas as pd
import numpy as np
import os
from tqdm import tqdm           # for loops
import re                       # regular expressions
from bs4 import BeautifulSoup   # web scraping

HACKBAY_FILE = r'C:\Users\vi04wecu\Desktop\Hackbay'
DATA_INPUT = r'D:\hackbay\data'
#DATA_INPUT = r'C:\Users\vi04wecu\Desktop\Hackbay\data'
DATA_OUTPUT = r'C:\Users\vi04wecu\Desktop\Hackbay\processed_data'

# Data extraction 
text_ = []
title_ = []
keywords_ = []
number_of_images=[]
url=[]
colors = []
number_of_images=[]
MEDIA_FORMATS = ['jpg','jpeg','png','svg','jfif','gif']
color_regex = r"#([A-Fa-f0-9]{6})"          # get color hex
# get gender, age, text, keyword, title, image names, url_id, color

#directory_lists = os.path.normpath("/content/drive/MyDrive/unzipped_hackbay")       # change / to \ in datapath

directory_lists = DATA_INPUT 
for directory in os.listdir(directory_lists):
  directory = os.path.join(directory_lists,directory)
  with tqdm(total = len(os.listdir(directory))) as pbar:             
        for fi in os.listdir(directory):
            file = os.path.join(directory_lists, directory, fi)
            if fi.endswith(".html"):
                with open(file,'r', encoding="ISO-8859-1") as f: #ISO-8859-1
                    a = f.read()
                    line = BeautifulSoup(a)
                    #parra = line.find('p').getText()
                    text, title, keywords = None, None, None

                    # Get possible colors
                    colors_found = re.findall(color_regex,a)
                    if colors_found!=[]:
                        colors_found = list(set(colors_found))
                        colors_found = ['#'+col for col in colors_found]
                        colors_found = ','.join(colors_found)
                    else:
                        colors_found = ''  
                    
                    # Find number of images
                    images_found = line.findAll('img')
                    number_of_images_to_append = len(images_found) if images_found!=[] else 0
                    
                    # Get text
                    try:
                        text = '|'.join([p.text for p in line.find('body').find_all('p')])
                        title = line.find('title').text
                        description_find = line.find("meta", {'name':"description"})
                        if description_find and 'content' in description_find:
                            title+=' '+description_find['content']
                            
                        og_title = line.find("meta",  {"property":"og:title"})
                        if og_title and 'content' in og_title:
                            title+=' '+og_title['content']
                        keywords_find = line.find("meta", {'name':"keywords"})
                        keywords = keywords_find['content'] if keywords_find and 'content' in keywords_find else None                 
                    except AttributeError:
                        if text is None and title is None and keywords is None:
                            continue       
                        else:
                            text = text if text is not None else ''
                            title = title if title is not None else ''
                            keywords = keywords if keywords is not None else ''
                            
                    text_.append(text)
                    title_.append(title)
                    keywords_.append(keywords)
                    url.append(fi.split('.html')[0])
                    colors.append(colors_found)
                    number_of_images.append(number_of_images_to_append)
        pbar.update(1)

assert len(text_) == len(title_) == len(keywords_) == len(url) == len(colors) == len(number_of_images)

df = pd.DataFrame({'text':text_,'title':title_,'keywords':keywords,'url_id':url,'colors':colors,'number_of_images':number_of_images})
df.to_csv(os.path.join(DATA_OUTPUT, 'hackbay_features.csv'),index=False,encoding='utf8')

# File with labels 
df_ = pd.read_csv(os.path.join(DATA_OUTPUT, 'hackbay_features.csv'),encoding='ISO-8859-1').fillna('')

train_df = pd.read_csv(os.path.join(HACKBAY_FILE,'train.csv'))
train_dataset = pd.merge(train_df,df_,how="inner", on="url_id")
train_dataset.to_excel(os.path.join(DATA_OUTPUT, 'hackbay_train_dataset.xlsx'),index=False)

# Working on test

test_df = pd.read_csv(os.path.join(HACKBAY_FILE,'test.csv'))
test_dataset = pd.merge(test_df,df_,how="left", on="url_id")
test_dataset.to_excel(os.path.join(DATA_OUTPUT, 'hackbay_test_dataset.xlsx'),index=False)