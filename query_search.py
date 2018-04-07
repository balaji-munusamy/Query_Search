
##############################################approach 1
#define all imports
from gensim.models import Word2Vec
import gensim
import pandas as pd
import numpy as np
import nltk
import csv
import string
from gensim.models import Phrases
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import os
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import PunktSentenceTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import TfidfVectorizer
lemmatizer = WordNetLemmatizer()
import collections
from collections import OrderedDict
import datetime
import time
import pickle
#import spacy
#nlp = spacy.load('en')  

generic_model = ""
use_genericmodel = True
    
#define all stopwords
def stopwords_list():
    stop_words_list = set(stopwords.words('english'))
#    stop_words_list.update(('and','a','so','arnt','this','when','It','many','so','cant','yes'))
#    stop_words_list.update(('no','these','these','please', 'let', 'know', 'cant', 'can', 'pls', 'u', 'abt', 'wht'))
    return stop_words_list

stop_words_list = stopwords_list()

sys_out_response = False


# In[2]:


###############################################preprocessing text functions
#function to process a sentenace - remove punctuation and stop words
def process_data(text, stopwords_remove= True):

    if(stopwords_remove):
        return [lemmatizer.lemmatize(word) for word in basic_tokenizer(text) if word not in stop_words_list if word not in string.punctuation]
    else:#without stopword removal
        return [lemmatizer.lemmatize(word) for word in basic_tokenizer(text) if word not in string.punctuation]
    
# Convert text to lower-case and strip punctuation/symbols from words
def normalize_text(text, stopwords_remove = True):
    norm_text = clean_text(text)
    resp = process_data(norm_text, stopwords_remove)
    return resp

# =============================================================================
def clean_text(text):
    norm_text = text.lower()
    # Replace breaks with spaces
#    norm_text = norm_text.replace('<br />', ' ')
    # Pad punctuation with spaces on both sides
    for char in ['.', '"', ',', '(', ')', '!', '?', ';', ':']:
        norm_text = norm_text.replace(char, ' ' + char + ' ')
        
        
    norm_text = norm_text.replace('\'', '')
    
    norm_text = norm_text.lower()
    return norm_text
# =============================================================================

#Very basic tokenizer: split the sentence into a list of tokens."""
def basic_tokenizer(sentence):
    import re
    _WORD_SPLIT = re.compile(b"([.,!?\"':;)(])")
    words = []
    for space_separated_fragment in sentence.split():
        #words.extend(re.split(_WORD_SPLIT, space_separated_fragment))
        words.append(space_separated_fragment)
    return [w for w in words if w]

def find_pos(word): 
    tagged = nltk.pos_tag(basic_tokenizer(word))
    return tagged[0][1]

        


# In[3]:


def load_data(input_file):
    questions_processed = []
    qsnt_proc = []
    questions = []
    answers = []
    bigram = gensim.models.Phrases()
    with open(input_file, "r") as sentencesfile:
        reader = csv.reader(map(lambda line:line,sentencesfile), delimiter = ",")
        for row in reader:
            qstn = str(row[5])
            ans = str(row[6])
            normalized_text = normalize_text(qstn, False)
            questions_processed.append(normalized_text)
            qsnt_proc.append(qstn)
#            qsnt_proc.append(clean_text(qstn))
            questions.append(qstn) #questions list
            answers.append(ans)   #answers list
    return list(bigram[questions_processed]), qsnt_proc, questions, answers


# In[4]:


# from sklearn.feature_extraction.text import TfidfVectorizer
def create_tfidfmodel(dataset):
    #import sklearn.feature_extraction.text.TfidfVectorizer
    tfidf_model = TfidfVectorizer(stop_words =  stop_words_list, ngram_range =(1,3))
    tfidf_model.fit(dataset)
    return tfidf_model

def tfidf_sentencevector(sentence, model):
    return model.transform(sentence)

#function to create word2vec model
def create_word2vecmodel(dataset, min_count=0,size=300, window=5, workers=4):
    return gensim.models.Word2Vec(dataset, min_count=0,size=300, window=5, workers=4)

def find_idf_score(model, word):
    response =1
    try:
        feat_nam = model.get_feature_names()
        ind = feat_nam.index(word) 
        idf = model.idf_
        response = idf[ind]
    except:
        #print('err', word)
        response = 1
    return response
 
    
###get frequency
def find_count_word(word, sentence): 
    import re
    #line = " I am having a very nice day am."
    count = len(re.findall(word, sentence))
    return (count)

###get frequency
def find_count_word_list(word, sentence_list): 
    return (sentence_list.count(word))

def find_scaling_factor(pos):
    scaling_factor = 0.25  # default value
    if(str(pos)=='NNPS' or str(pos)=='NNP'):
        scaling_factor = 100
    elif(str(pos)=='NN' or str(pos)=='NNS'):
        scaling_factor = 90
    elif(pos.find('VB')!=-1):
        scaling_factor = 50
    elif(str(pos)=='PRP' or str(pos)=='PRP$'):
        scaling_factor = 0
    return scaling_factor

#function to find the sentence vector using word2vec average of all words
def words_to_sentencevector(sentence, model, generic_model, model_qstns_tfidf, num_features):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    pos_tagged = nltk.pos_tag(basic_tokenizer(' '.join(sentence)))
    #print('words_to_sentencevector called')
    counter = 0
    for word in sentence:

        pos = pos_tagged[counter][1] #find_pos(word) 
        scaling_factor = find_scaling_factor(pos)
        
        idf_score = find_idf_score(model_qstns_tfidf, str(word))
        word_count = find_count_word_list(str(word), sentence)
        tfidf_score =  idf_score * word_count
        
        update_val = scaling_factor * tfidf_score
        if(use_genericmodel):
            if word in generic_model.wv.vocab:
                temp_vect = generic_model[word] 
                featureVec = np.add(featureVec, np.multiply(update_val, temp_vect))
                nwords = nwords+1
            elif word in model.wv.vocab:
                temp_vect = model[word] 
                featureVec = np.add(featureVec, np.multiply(update_val, temp_vect))   
                nwords = nwords+1
            else:
                print('word not in dictionary', word)
        else:
            if word in model.wv.vocab:
                temp_vect = model[word] 
                featureVec = np.add(featureVec, np.multiply(update_val, temp_vect))
                nwords = nwords+1
            elif word in generic_model.wv.vocab:
                temp_vect = generic_model[word] 
                featureVec = np.add(featureVec, np.multiply(update_val, temp_vect))   
                nwords = nwords+1
            else:
                print('word not in dictionary', word) 
        counter = counter+1
    if(nwords>0):
        featureVec = np.divide(featureVec, nwords)
        #featureVec_ret = np.mean(featureVec)
        
    return featureVec


# In[5]:


#functions to find cosine similarity
def find_cosine_similarity(vector1, vector2):
    print('cosine similarity called')
    from numpy.linalg import norm
    try:
        return np.dot(vector1, vector2)/(norm(vector1)*norm(vector1)) 
    except:
        return 0

def find_cosine_similarity_sp(vector1, vector2):
    #import scipy as sp
    try:
        return (1-cosine(vector1,vector2))
    except:
        return 0

def find_cosine_similarity_scikit(vector1, vector2):
    #import scipy as sp
    try:
        return (cosine_similarity(vector1, vector2))
    except:
        return 0


# In[6]:


####functions to create vector list of all qstns
def createw2v_sentencevec_list(sentence_list,word_model,generic_model, model_qstns_tfidf):
    sentnc_vectr_list = []
    for sentence in sentence_list:
        normalized_sent = normalize_text(sentence, True)
        #get average vector for sentence 2(word2vec)
        sentence_wvvector = words_to_sentencevector(normalized_sent, model=word_model, generic_model= generic_model, model_qstns_tfidf = model_qstns_tfidf, num_features=300)         
        sentnc_vectr_list.append(sentence_wvvector)
    return sentnc_vectr_list
 
def createtfidf_sentencevec_list(sentence_list,model_qstns_tfidf):
    sentnc_vectr_list = []
    for sentence in sentence_list:
        normalized_sent = str(normalize_text(sentence, True))
        sentence_tfidfvector =  tfidf_sentencevector([normalized_sent], model_qstns_tfidf)
        sentnc_vectr_list.append(sentence_tfidfvector)
    return sentnc_vectr_list


# In[7]:


#find matching sentence word2vec
def find_matching_sentence(sentence_1, sentence_list_vectors, word_model, generic_model, model_qstns_tfidf):

    normalized_sent_1 = normalize_text(sentence_1, True)
    #get average vector for query (word2vec)
    sentence_1_wvvector = words_to_sentencevector(normalized_sent_1, model=word_model, generic_model= generic_model, model_qstns_tfidf= model_qstns_tfidf, num_features=300)
    sentence_1_wvvector_unique = list(set(sentence_1_wvvector))
    best_match_score = 0
    sentence_count=0
    best_match_index =-1

    #compare sentence1 against all sentences in sentence_list
    similarty_dict = {}

    for sentence_vector in sentence_list_vectors:
        try:
            if(len(sentence_1_wvvector_unique)==1 and sentence_1_wvvector_unique[0]==0):
                sen1_sen2_similarity =  0     
            else:
                sen1_sen2_similarity =  find_cosine_similarity_sp(sentence_1_wvvector,sentence_vector)
                
            similarty_dict[sentence_count] = sen1_sen2_similarity
        except:
            print('error in word2vec similarity')
        sentence_count=sentence_count+1
        
    return similarty_dict


# In[8]:


#find matching sentence tfidf
def find_matching_sentence_tfidf(sentence_1, sentence_list_vectors, model_qstns_tfidf):

    normalized_sent_1 = str(normalize_text(sentence_1, True))
    sentence_1_tfidfvector = tfidf_sentencevector([normalized_sent_1], model_qstns_tfidf)
    best_match_score = 0
    sentence_count=0
    best_match_index =-1
    
    #compare sentence1 against all sentences in sentence_list
    similarty_dict = {}
    for sentence_vector in sentence_list_vectors:
        try:
            #get  vector for sentence 2
            sen1_sen2_similarity =  find_cosine_similarity_scikit(sentence_1_tfidfvector,sentence_vector)[0][0]
            similarty_dict[sentence_count] = sen1_sen2_similarity
        except:
            print('error in tfidf similarity')
        sentence_count=sentence_count+1
        
    return similarty_dict


# In[9]:


## function to generate response 
def generate_response(query, w2v_sent_list, tfidif_sent_list, model_qstns, generic_mode,model_qstns_tfidf): 
    response_followup = False
    dict_similarity_w2v = find_matching_sentence(query, w2v_sent_list, model_qstns, generic_model, model_qstns_tfidf)
    dict_similarity_tfidf = find_matching_sentence_tfidf(query, tfidif_sent_list, model_qstns_tfidf)
    best_indexes, best_scores = get_best_responses(dict_similarity_w2v, dict_similarity_tfidf)
    response = ""
    if(best_scores[0]>=0.85):
        print("Response:", ans[best_indexes[0]])
        response = ans[best_indexes[0]]
#    elif((best_scores[0] < 0.85 and best_scores[0] > 0.2)):
#        #text = 'did you mean "' + qstns[best_match_index] + '"?'
#        print('Would you like information on any of the following?:')
#        i=1
#        resp  = '[' + str(i)  + "." +  ']' +  qstns[best_indexes[0]]
#        print(resp)
#        if(best_scores[1] > 0.5):
#            i =i +1
#            resp = '[' + str(i)  + "." +  ']' + qstns[best_indexes[1]]
#            print(resp)
#        if(best_scores[2] > 0.5):
#            i = i + 1
#            resp = '[' + str(i) + "." +  ']' + qstns[best_indexes[2]]
#            print(resp)
#        print('Please select the applicable item (e.g. 1,2...). Select NA in case none of the choices are applicable.')
#        response_followup = True
        
        
    else:
        response = "Sorry I am unable to understand this query. Could you please elaborate?"
        print("Sorry I am unable to understand this query. Could you please elaborate?")

    return best_indexes, best_scores, response_followup, response

#function to find best responses
def get_best_responses(dict_similarity_w2v, dict_similarity_tfidf):
    responses_no = 3
    scores_sorted_w2v = sorted(dict_similarity_w2v.values(), reverse=True)
    scores_sorted_tfidf = sorted(dict_similarity_tfidf.values(), reverse=True) 
    best_scores_indexes = []
    best_scores = []

    i_w2v=0
    j_w2v = 0
    count =0

    while(len(best_scores_indexes) < responses_no):
        
        wv_score = scores_sorted_w2v[i_w2v]
        tfidf_score = scores_sorted_tfidf[j_w2v]
        if('nan' == str(wv_score)):
            wv_score = 0

        if(tfidf_score == 0 and wv_score == 0):
            best_scores_indexes.append(0)
            best_scores.append(0)
            #print('scores added to ', len(best_scores_indexes))
            break
        if(wv_score>=tfidf_score):
            index_val_wv = list(dict_similarity_w2v.keys())[list(dict_similarity_w2v.values()).index(wv_score)]
            if(index_val_wv not in best_scores_indexes):
                best_scores_indexes.append(index_val_wv)
                best_scores.append(wv_score)
                #print('scores added to ', len(best_scores_indexes))
                #print('wv score added')
            i_w2v =i_w2v+1
        else:
            index_val_tf = list(dict_similarity_tfidf.keys())[list(dict_similarity_tfidf.values()).index(tfidf_score)]
            if(index_val_tf not in best_scores_indexes):
                best_scores_indexes.append(index_val_tf)
                best_scores.append(tfidf_score)
                #print('scores added to ', len(best_scores_indexes))
                #print('tfidf score added')
            j_w2v = j_w2v +1
        count = count +1
    return best_scores_indexes, best_scores





# In[ ]:


def update_training_data(query):
    if(query.strip().upper() != 'NA' and query.isalpha()): 
        qstns.append(prev_query)
        ans.append(ans[best_indexes[int(query)-1]])
        pro_qsnts_norm.append(prev_query)
        model_qstns_tfidf = create_tfidfmodel(pro_qsnts_norm)
        tfidif_sent_list = createtfidf_sentencevec_list(qstns,model_qstns_tfidf)
        print(ans)
        print('Query added to list. Thank you!')
    else:
        print('Query ignored. Thank you!') 
        
        
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("Start time is ", st)
        
processed_qstns, pro_qsnts_norm, qstns, ans = load_data('agri-dataset.csv')

#os.chdir("D:\\Syntel\\Work\Hackathon\\Nasscom-hackathon\\dataset\\")

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("Load data is ", st)

generic_w2v_model = 'GoogleNews-vectors-negative300.bin'
#generic_w2v_model = 'knowledge-vectors-skipgram1000.bin'
generic_model = gensim.models.KeyedVectors.load_word2vec_format(generic_w2v_model, binary=True, limit=1000000000)
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("Generic model is ", st)

model_qstns = gensim.models.Word2Vec.load('w2v.model')
#model_qstns = create_word2vecmodel(processed_qstns)
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("W2V model is ", st)
#model_qstns.save('w2v.model')


#training tfidf model
model_qstns_tfidf = pickle.load(open("tfidf_model.pickle", "rb"))
#model_qstns_tfidf = create_tfidfmodel(pro_qsnts_norm)
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("TFIDF model is ", st)
#pickle.dump(model_qstns_tfidf, open('tfidf_model.pickle', 'wb'))
#model for pos tagging

w2v_sent_list = createw2v_sentencevec_list(qstns,model_qstns,generic_model, model_qstns_tfidf)
ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("W2V sentence list is ", st)
tfidif_sent_list = createtfidf_sentencevec_list(qstns,model_qstns_tfidf)

ts = time.time()
st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
print("W2V sentence list is ", st)

count = 1
response_followup = False
prev_query = ""
print('Hi, How may I help you?')

#############################################query interface
        

def query_interface(query):
	print('Query: ', query)
    best_indexes, best_scores, response_followup, response = generate_response(query, w2v_sent_list, tfidif_sent_list, model_qstns, generic_model, model_qstns_tfidf)
    return response;
    
    

