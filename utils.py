import numpy as np
import pandas as pd

# NLP libraries
import string
import contractions
import nltk.tag as tag
from nltk.tokenize import word_tokenize
from gensim.models import phrases
from gensim.models import CoherenceModel
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from gensim.models import wrappers

# Deep learning
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Input, Flatten, LSTM, Embedding, SpatialDropout1D
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences



# Converting parts to wordnet format 
def get_wordnet_pos(tag):
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

    
def preprocess_text(df):
    """
    Preprocess training data and return clean df"
    """
    # Removing contractions and text tokenization
    df['no_contract'] = df['concat'].apply(lambda x: [contractions.fix(word) for word in x.split()])
    
    # Tokenizing text
    df['no_contract_str'] = [' '.join(map(str, l)) for l in df['no_contract']]
    df['tokenized'] = df['no_contract_str'].apply(word_tokenize)
    
    # Convetrting words to lowercase and deleting short words
    df['lower'] = df['tokenized'].apply(lambda x: [word.lower() for word in x  if len(word) >= 3])
    
    # Removing punctuation
    punc = string.punctuation
    df['no_punc'] = df['lower'].apply(lambda x: [word for word in x if word not in punc])
    
    # Removing stopwords 
    stop_words = set(stopwords.words('english'))
    df['stopwords_removed'] = df['no_punc'].apply(lambda x: [word for word in x if word not in stop_words])
    
    # Make bigrams & trigrams
    bigram = phrases.Phrases(df['stopwords_removed'], min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = phrases.Phrases(bigram[df['stopwords_removed']], threshold=100)  
    bigram_mod = phrases.Phraser(bigram)
    trigram_mod = phrases.Phraser(trigram)
    
    df['trigram'] = [trigram_mod[bigram_mod[doc]] for doc in df['stopwords_removed']]
    
    # Applying speech tags
    df['pos_tags'] = df['trigram'].apply(tag.pos_tag)
    
    # Converting parts to wordnet format
    df['wordnet_pos'] = df['pos_tags'].apply(lambda x: [(word, get_wordnet_pos(pos_tag)) for (word, pos_tag) in x])
    
    #Lemmatization
    wnl = WordNetLemmatizer()
    df['lemmatized'] = df['wordnet_pos'].apply(lambda x: [wnl.lemmatize(word, tag) for word, tag in x])
    
    df['lemma_str'] = [' '.join(map(str,l)) for l in df['lemmatized']]
    
    df.to_csv('./data/clean_joined.csv')
    
    return df


def compute_coherence(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    """
    mallet_path = './mallet-2.0.8/bin/mallet'
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = wrappers.LdaMallet(mallet_path, random_seed=42, corpus=corpus, num_topics=num_topics, id2word=dictionary)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


def plot_coefficients(classifier, feature_names, top_features=15):
    if type(best_svm.named_steps['clf']).__name__ == 'SGDClassifier':
        coef = classifier.coef_.ravel()
    elif type(best_svm.named_steps['clf']).__name__ == 'MultinomialNB':
        coef = classifier.feature_log_prob_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ["red" if c < 0 else "blue" for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 2 * top_features), feature_names[top_coefficients], rotation=60, ha="right")
    plt.show()

    
def prepare_data(train_str, test_str, model_type, mode='freq', num_words=80000, max_len=256):
    # create the tokenizer
    tokenizer = Tokenizer(num_words=num_words)
    # fit the tokenizer on the documents
    tokenizer.fit_on_texts(train_str)
    if model_type == 'nbow':
        # encode training data set
        X_train = tokenizer.texts_to_matrix(train_str, mode=mode)
        # encode test data set
        X_test = tokenizer.texts_to_matrix(test_str, mode=mode)
    elif model_type == 'lstm':
        # encode training data set
        X_train = pad_sequences(tokenizer.texts_to_sequences(train_str), maxlen=max_len)
        # encode test data set
        X_test = pad_sequences(tokenizer.texts_to_sequences(test_str), maxlen=max_len)
    return X_train, X_test


def compile_model(input_shape, learning_rate, model_type, num_words=80000, embedding_dim=128):
    model = Sequential()
    if model_type == 'nbow':
        
        model.add(Dense(16, input_shape=(input_shape,), activation='relu'))
        model.add(Dropout(0.8))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    
    elif model_type == 'lstm':
        model.add(Embedding(num_words, embedding_dim, input_length=input_shape))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(128, activation='tanh', dropout=0.2, 
                       recurrent_activation='sigmoid', recurrent_dropout=0.2, 
                       unroll=False, use_bias=True))
        model.add(Dense(1, activation='sigmoid'))

    
    # compile network
    _learning_rate = learning_rate
    opt = Adam(learning_rate=_learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model