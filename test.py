import pandas as pd
from eval import preprocess_text
from utils import remove_stopwords 
from functools import reduce

def clean_title(text):
    text= text.lower()
    text = preprocess_text(text)
    text = remove_stopwords(text)
    tokens = text.split(' ')
    tokens = list(set(tokens))
    return '|'.join(tokens)





df = pd.read_excel('hackbay_clean_dataset.xlsx').fillna('')
df['colors'] = df['colors'].apply(lambda x: x+',')
df['title_cleaned'] = df['title_cleaned'].astype(str)
df['title_cleaned'] = df['title_cleaned'].apply(lambda x: x+'|')

grouped_df =df.groupby(['age','gender'])
color_agg  = pd.DataFrame(grouped_df.agg(colors = ('colors','sum')).to_records())

keyword_agg = pd.DataFrame(grouped_df.agg(keywords_ = ('title_cleaned','sum')).to_records())
keyword_agg['keywords_'] = keyword_agg['keywords_'].astype(str)
image_agg  = pd.DataFrame(grouped_df['number_of_images'].agg([lambda x: int(x.quantile(0.45)),lambda x: int(x.quantile(0.55)) ]).to_records())
image_agg = image_agg.rename({'<lambda_0>':'0.45','<lambda_1>':'0.55'},axis='columns')

data_frames = [color_agg,keyword_agg,image_agg]
df_merged = reduce(lambda left,right: pd.merge(left,right,on=['age','gender'],
                                            how='inner'), data_frames)

df_merged.to_excel('hackbay_recommendations.xlsx',index=False)


print('ALL DONE')
