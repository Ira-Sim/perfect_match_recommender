from nltk.corpus import stopwords
from PIL import Image
stop_words = stopwords.words('german') +  stopwords.words('english')

def remove_stopwords(text):
    # Given a text, remove the stopwords in them
    token = text.split()
    return ' '.join([w for w in token if not w in stop_words])

def create_mean(data):
    mean_value = data.groupby(['age','gender'])['image_text_ratio'].transform('mean')
    return mean_value

def output_mean(data):
    mean_value_whole_data = create_mean(data)

    mean_value_whole_data = mean_value_whole_data * 100

    str_mean_value_whole_data = mean_value_whole_data.astype(str) + '%'
    return str_mean_value_whole_data

def output_mean_for_whole_dataset(dataset):
    mean_of_the_whole_dataset= dataset['image_text_ratio'].agg('mean')
    mean_of_the_whole_dataset = mean_of_the_whole_dataset.astype(str) + '%'
    return mean_of_the_whole_dataset
