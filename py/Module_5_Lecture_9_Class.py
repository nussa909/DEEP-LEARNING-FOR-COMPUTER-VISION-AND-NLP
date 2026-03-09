# %%
import re
import string
import itertools
from collections import Counter

import pandas as pd
import numpy as np

# import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

import spacy

from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from tqdm.auto import tqdm
tqdm.pandas()

import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv('../data/Module_5_Lecture_1_Class_amazon_product_reviews.csv', index_col='Id')

# %%
# Dataset preview

df.head(3)

# %%
# Basic dataset info 

df.info()

# %%
# For this lesson, let's ignore neutral reviews
# We'll consider reviews with a score 3 as neutral

df = df.loc[df['Score']!=3]

df.shape

# %%
df['sentiment'] = [1 if score in [4, 5] else 0 for score in df['Score']]

# %%
# Number of identical records

df.duplicated().sum()

# %%
# Droping duplicated records

df = df.drop_duplicates().reset_index(drop=True)

# Number of unique records

df.shape

# %%
# Checking for the identical reviews of different versions of the same product

df.groupby(['UserId', 'Time', 'Text']).count().sort_values('ProductId', ascending=False).head(10)

# %%
search_text = "I have two cats, one 6 and one 2 years old. Both are indoor cats in excellent health. I saw the negative review and talked to my vet about it. I've also asked a number of veterinary professionals what to feed my cats and they all answer the same thing: Science Diet. Sure, you'll see stories of how one person's cat had issues, but even if that's 100% true, it's 1 case out of millions. Science and fact aren't based on someone's experience.<br /><br />So my point is, I love my cats and I'm very concerned about their health. I trust people who actually have medical degrees and experience with a wide range of animals. My only caution is do not fall for some hype or scare tactic that recommends some unproven or untested food or some fad diet for your pet. Don't listen to me, don't listen to the negative review. ASK YOUR VET what they recommend, and follow their instructions. My guess is you'll end up buying the Science Diet anyhow."
duplicates_example = df.loc[
    (df['UserId']=='A36JDIN9RAAIEC') &
    (df['Time']==1292976000) &
    (df['Text']==search_text)
]

duplicates_example

# %%
# Droping the same reviews

df = df.drop_duplicates(subset={"UserId", "Time","Text"})

# Final size 

df.shape

# %%
# Contractions. Source http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he's": "he is",
"how'd": "how did",
"how'll": "how will",
"how's": "how is",
"i'd": "i would",
"i'll": "i will",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'll": "it will",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"must've": "must have",
"mustn't": "must not",
"needn't": "need not",
"oughtn't": "ought not",
"shan't": "shall not",
"sha'n't": "shall not",
"she'd": "she would",
"she'll": "she will",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"that'd": "that would",
"that's": "that is",
"there'd": "there had",
"there's": "there is",
"they'd": "they would",
"they'll": "they will",
"they're": "they are",
"they've": "they have",
"wasn't": "was not",
"we'd": "we would",
"we'll": "we will",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"where'd": "where did",
"where's": "where is",
"who'll": "who will",
"who's": "who is",
"won't": "will not",
"wouldn't": "would not",
"you'd": "you would",
"you'll": "you will",
"you're": "you are"
}

# %%
# get stop-words from the nltk library
# using set to make words search faster

stop_words = set(stopwords.words('english')).union({'also', 'would', 'much', 'many'})

# %%
negations = {
    'aren',
    "aren't",
    'couldn',
    "couldn't",
    'didn',
    "didn't",
    'doesn',
    "doesn't",
    'don',
    "don't",
    'hadn',
    "hadn't",
    'hasn',
    "hasn't",
    'haven',
    "haven't",
    'isn',
    "isn't",
    'mightn',
    "mightn't",
    'mustn',
    "mustn't",
    'needn',
    "needn't",
    'no',
    'nor',
    'not',
    'shan',
    "shan't",
    'shouldn',
    "shouldn't",
    'wasn',
    "wasn't",
    'weren',
    "weren't",
    'won',
    "won't",
    'wouldn',
    "wouldn't"
}

# %%
# removing negations from the stop-words list

stop_words = stop_words.difference(negations)

# %%
stemmer = PorterStemmer()

# %%
nlp = spacy.load("en_core_web_sm", disable = ['parser','ner'])

# %%
# function to clean text
def normalize_text(raw_review):
    
    # Remove html tags
    text = re.sub("<[^>]*>", " ", raw_review) # match <> and everything in between. [^>] - match everything except >
    
    # Remove emails
    text = re.sub("\S*@\S*[\s]+", " ", text) # match non-whitespace characters, @ and a whitespaces in the end
    
    # remove links
    text = re.sub("https?:\/\/.*?[\s]+", " ", text) # match http, s - zero or once, //, 
                                                    # any char 0-unlimited, whitespaces in the end
        
     # Convert to lower case, split into individual words
    text = text.lower().split()
    
    # Replace contractions with their full versions
    text = [contractions.get(word) if word in contractions else word 
            for word in text]
   
    # Re-splitting for the correct stop-words extraction
    text = " ".join(text).split()    
    
    # Remove stop words
    text = [word for word in text if not word in stop_words]

    text = " ".join(text)
    
    # Remove non-letters        
    text = re.sub("[^a-zA-Z' ]", "", text) # match everything except letters and '


    # Stem words. Need to define porter stemmer above
    # text = [stemmer.stem(word) for word in text.split()]

    # Lemmatize words. Need to define lemmatizer above
    doc = nlp(text)
    text = " ".join([token.lemma_ for token in doc if len(token.lemma_) > 1 ])
    
    # Remove excesive whitespaces
    text = re.sub("[\s]+", " ", text)    
    
    # Join the words back into one string separated by space, and return the result.
    return text

# %%
text = 'On a quest for the perfedc1112t,,, !!!! <br />%%2%% popcorn to compliment the Whirley Pop.  Don\'t get older, I\'m beginning to appreciate the more "natural" popcorn varieties, and I suppose that\'s what attracted me to the Arrowhead Mills Organic Yellow Popcorn.<br /> <br />I\'m no "organic" food expert.  I just wanted some good tasting popcorn.  And, I feel like that\'s what I got.  Using the Whirley Pop, with a very small amount of oil, I\'ve had great results.'

print('Original text', text, '#'*30, sep='\n\n')
print('\nNormalized text', normalize_text(text), sep='\n\n')

# %%
# Slicing dataset for demonstrative purposes

df = df.groupby('sentiment').sample(2500, random_state=42)

# Note: without resetting an index we slice over the original Id`s
df.shape

# %%
df['text_normalized'] = df['Text'].progress_apply(normalize_text)

# %%
train_idxs = df.sample(frac=0.8, random_state=42).index
test_idxs = [idx for idx in df.index if idx not in train_idxs]

# %%
X_train = df.loc[train_idxs, 'text_normalized']
X_test = df.loc[test_idxs, 'text_normalized']

y_train = df.loc[train_idxs, 'sentiment']
y_test = df.loc[test_idxs, 'sentiment']

# %%
# Creating and training a CountVectorizer object 

vect = CountVectorizer().fit(X_train)

len(vect.vocabulary_)

# %%
# features examples

vect.get_feature_names_out()[:5]

# %%
# transform the documents in the training data to a document-term matrix

X_train_vectorized = vect.transform(X_train)
X_train_vectorized.shape

# %%
# Resulted features representation is a sparse matrix

X_train_vectorized

# %%
model = LogisticRegression(random_state=42)
model.fit(X_train_vectorized, y_train)

# %%
predictions = model.predict(vect.transform(X_test))

print('AUC: ', roc_auc_score(y_test, predictions))

# %%
def get_preds(text_column, algorithm, ngrams=(1,1)):
    
    X_train = df.loc[train_idxs, text_column]
    X_test = df.loc[test_idxs, text_column]

    y_train = df.loc[train_idxs, 'sentiment']
    y_test = df.loc[test_idxs, 'sentiment']
    
    if algorithm == 'cv':
        vect = CountVectorizer(ngram_range=ngrams).fit(X_train)
    elif algorithm == 'tfidf':
        vect = TfidfVectorizer(ngram_range=ngrams).fit(X_train)
    else:
        raise ValueError('Select correct algorithm: `cv` or `tfidf`')
            
    print('Vocabulary length: ', len(vect.vocabulary_))
    
    # transform the documents in the training data to a document-term matrix

    X_train_vectorized = vect.transform(X_train)
    print('Document-term matrix shape:', X_train_vectorized.shape)
    
    model = LogisticRegression(random_state=42)
    model.fit(X_train_vectorized, y_train)
    
    predictions = model.predict(vect.transform(X_test))

    print('AUC: ', roc_auc_score(y_test, predictions))

# %%
get_preds('Text', 'cv')

# %%
get_preds('text_normalized', 'tfidf')

# %%
get_preds('Text', 'tfidf')

# %%
get_preds('text_normalized', 'cv', (1,2))

# %%
get_preds('text_normalized', 'tfidf', (1,2))

# %%
get_preds('text_normalized', 'cv', (2,2))

# %%
get_preds('Text', 'cv', (2,2))

# %%
get_preds('Text', 'tfidf', (2,2))

# %%


# %%


# %%
