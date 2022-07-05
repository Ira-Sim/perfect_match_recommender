import streamlit as st
import os
import matplotlib.pyplot as plt
from eval import predict,PerfectMatch,AGE_TO_INDEX,GENDER_TO_INDEX,load_checkpoint
from transformers import BertTokenizer, BertModel
import torch 
import torch.optim as optim
import requests
from bs4 import BeautifulSoup
import pandas as pd
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from PIL import Image
import numpy as np

def get_image(img_path):
    image = Image.open(img_path)
    return image

st.set_page_config(
    page_title="Perfect Match?",
    layout="wide",
    initial_sidebar_state="auto",
)


st.sidebar.image(get_image('image_logo.png'),width=300)
st.sidebar.markdown(
    """
Using deep learning, we leverage millions of other websites to bring you useful recommendations to make your online business succesful! 
"""
)

device = torch.device('cuda' if True and torch.cuda.is_available() else 'cpu')
 
@st.cache
def load_params():
    model = PerfectMatch().to(device)
    model_path = 'perfect_match_model.pt'
    if os.path.exists(model_path):
        optimizer = optim.Adam(model.parameters(), lr=0.0001)
        model,optimizer = load_checkpoint(model_path, model, optimizer)

    model = model.to(device)
    # Define tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    
    # Load (and possibly transform) our dataset which will be used for making recommendations
    recommendation_df = pd.read_excel('hackbay_recommendations.xlsx')
    return model,tokenizer,recommendation_df



def plot_pie(labels,sizes):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    max_size = max(sizes)
    explode = [0.1 if i==max_size else 0 for i in sizes]
    fig1, ax1 = plt.subplots(figsize=(4,3))
    ax1.pie(sizes, explode=explode, labels=None, autopct='%.0f%%',
             startangle=90,colors=['#6da7cc', '#ffb58a'],  radius=1, pctdistance = 0.85, textprops={'color':"w", 'size' : '8'})
    
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.ylabel("")
    my_circle=plt.Circle( (0,0), 0.7, color='white')
    p=plt.gcf()
    p.gca().add_artist(my_circle)
    ax1.legend(labels = ['weiblich', 'männlich'],loc = 8, bbox_to_anchor=(0.5,0.4), frameon=False, fontsize = 8, labelcolor = 'linecolor')
    # legend below  bbox_to_anchor=(0.5,-0.2)
    #legend.get_frame().set_facecolor('none')
    plt.tight_layout()  
    return fig1


def col_pale(color_pale): 
    fig, ax = plt.subplots(figsize=(6, 1)) 
    x = list(range(len(color_pale)))
    h = np.ones(len(color_pale)).tolist()
    ax.bar(x,h, color=color_pale)
    plt.xticks([])
    plt.yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor("white")
    plt.tight_layout()
    return fig

def plot_bar(labels,sizes):
    #new_labels = ['10 bis 13 Jahre' , '14 bis 15 Jahre' , '16 bis 17 Jahre',  
    # '18 bis 19 Jahre' , '20 bis 24 Jahre', 
    # '25 bis 29 Jahre', '30 bis 34 Jahre', '35 bis 39 Jahre',
    # '40 bis 44 Jahre','45 bis 49 Jahre' , '50 bis 54 Jahre',
    # '55 bis 59 Jahre', '60 bis 64 Jahre', '65 bis 69 Jahre',
    # '70 bis 74 Jahre',   '75 und mehr Jahre']
    new_labels = ['10 bis 13' , '14 bis 15' , '16 bis 17',  
    '18 bis 19' , '20 bis 24', 
    '25 bis 29', '30 bis 34', '35 bis 39',
    '40 bis 44','45 bis 49' , '50 bis 54',
    '55 bis 59', '60 bis 64', '65 bis 69',
    '70 bis 74',   '75 und mehr']
    
    new_sizes=[sizes[new_labels.index(m)] for m in new_labels ]
    
    bar_df =pd.DataFrame({'age':new_labels,'probability':new_sizes}) 
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(5, 3)) 
    sns.barplot(x='age', y ='probability',data=bar_df, palette="Blues_d", ci=None)
    #ax.set_xlabel('age', fontsize=8)
    ax.set_xlabel(None)
    ax.set_ylabel('Probability', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_facecolor("white")
    plt.xticks(fontsize=8, rotation=90)
    plt.yticks(fontsize=8)
    plt.tight_layout()  
    return fig


def get_word_cloud(text):
    # lower max_font_size, change the maximum number of word and lighten the background:
    fig, ax = plt.subplots(figsize=(7, 4)) 
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    return fig
model, tokenizer,recommendation_df = load_params()     
text=''  
col_1, col_2, = st.columns([12,12])
main_url =st.sidebar.text_input(label="Type in your URL to know the audience that visits your website:").strip()

if main_url!='':
    # Make a GET request to fetch the raw HTML content
    html_content = requests.get(main_url).text

    # Parse the html content
    line = BeautifulSoup(html_content, "lxml")
    text = ''
    try:
        text = ' '.join([p.text for p in line.find('body').find_all('p')])
        title = line.find('title').text
        description_find = line.find("meta", {'name':"description"})
        if description_find and 'content' in description_find:
            title+=' '+description_find['content']
            text+=' '+title
    except AttributeError:
        text=''
        #st.sidebar.write('Please enter a valid website')
        
    if text!='':
        prediction = predict(text,model,tokenizer,distribution=True)
        
        fig_age = plot_bar(prediction['age']['labels'],prediction['age']['probs'])
        
        fig_gender = plot_pie(prediction['gender']['labels'],prediction['gender']['probs'])
      
        col_1.markdown("#### Your predicted audience profile")
        col_1.markdown("<h6 style='text-align: center; '>Age distribution</h6>", unsafe_allow_html=True) 
        # fig_age.savefig('age.png')  #Ira
        # image = Image.open('age.png')       # Ira
        # col_1.image(image, width=500)      # Ira
        col_1.pyplot(fig_age)
        col_1.markdown("""---""")
        col_1.markdown("<h6 style='text-align: center;'>Gender distribution</h6>", unsafe_allow_html=True)
        # fig_gender.savefig('gender.png')  #Ira
        # image = Image.open('gender.png')       # Ira
        # col_1.image(image, width=500)      # Ira
        col_1.pyplot(fig_gender)
st.sidebar.markdown("""---""")
st.sidebar.markdown("### Your Target Audience")
option_age = st.sidebar.selectbox(
    'Age',['Choose Age']+list(AGE_TO_INDEX.keys())
    )
option_gender = st.sidebar.selectbox(
    'Gender',['Choose Gender']+list(GENDER_TO_INDEX.keys())
    )
if option_age!='Choose Age' and option_gender!='Choose Gender':
    our_recommendation = recommendation_df[(recommendation_df['gender']==option_gender) & (recommendation_df['age']==option_age)]
    colors_recommend = [c_ for c_ in our_recommendation['colors'].values.tolist()]
    colors_recommend_ = [c.split(',') for c in colors_recommend if c is not np.nan]
    colors_recommend = [l for m in colors_recommend_ for l in m]
    color_pal = pd.Series(colors_recommend).value_counts()[3:8].index.tolist()

    keywords_recommend = [c_ for c_ in our_recommendation['keywords_'].values.tolist()]
    keywords_recommend = [c.split('|') for c in keywords_recommend]
    keywords_recommend = [l for m in keywords_recommend for l in m]
    col_2.markdown("#### Our recommendations to meet your target audience")
    col_2.markdown("#### Keyword/SEO recommendations")
    col_2.markdown("> Based on your target age and gender group, your website needs to have the following keywords")
    col_2.pyplot(get_word_cloud(' '.join(keywords_recommend)))
    col_2.markdown("""---""")

    images_recommend_up = [c_ for c_ in our_recommendation['0.45'].values.tolist()] 
    images_recommend_down = [c_ for c_ in our_recommendation['0.55'].values.tolist()]  
    col_2.markdown("#### Image recommendations")
    col_2.markdown(f"> Our analysis also suggests having __{images_recommend_up[0]}__ to __{images_recommend_down[0]}__ images in your web content.")  
    
    # Colors palette 
    colors_fig = col_pale(color_pal)
    col_2.markdown("""---""")
    col_2.markdown("#### Color recommendations")
    col_2.markdown("> Our analyses reveals the following colours on your website are better suited for your target audience")
    col_2.pyplot(colors_fig)








