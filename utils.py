from nltk.tokenize import word_tokenize
import re
import collections
import pickle
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from gensim.test.utils import get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec


train_article_path = "sumdata/train/train.article.txt"
train_title_path = "sumdata/train/train.title.txt"
valid_article_path = "sumdata/train/valid.article.filter.txt"
valid_title_path = "sumdata/train/valid.title.filter.txt"


def clean_str(sentence):
    sentence = re.sub("[#.]+", "#", sentence)
    return sentence


def get_text_list(data_path, toy):
    with open (data_path, "r", encoding="utf-8") as f:
        if not toy:
            return [clean_str(x.strip()) for x in f.readlines()]
        else:
            return [clean_str(x.strip()) for x in f.readlines()][:50000]


def build_dict(step, toy=False):
    if step == "train":
        train_article_list = get_text_list(train_article_path, toy)
        train_title_list = get_text_list(train_title_path, toy)

        words = list()
        for sentence in train_article_list + train_title_list:
            for word in word_tokenize(sentence):
                words.append(word)

        word_counter = collections.Counter(words).most_common()
        word_dict = dict()
        word_dict["<padding>"] = 0
        word_dict["<unk>"] = 1
        word_dict["<s>"] = 2
        word_dict["</s>"] = 3
        for word, _ in word_counter:
            word_dict[word] = len(word_dict)

        with open("word_dict.pickle", "wb") as f:
            pickle.dump(word_dict, f)

    elif step == "valid":
        with open("word_dict.pickle", "rb") as f:
            word_dict = pickle.load(f)

    reversed_dict = dict(zip(word_dict.values(), word_dict.keys()))

    article_max_len = 50
    summary_max_len = 15

    return word_dict, reversed_dict, article_max_len, summary_max_len


def build_dataset(step, word_dict, article_max_len, summary_max_len, toy=False):
    if step == "train":
        article_list = get_text_list(train_article_path, toy)
        title_list = get_text_list(train_title_path, toy)
    elif step == "valid":
        article_list = get_text_list(valid_article_path, toy)
    else:
        raise NotImplementedError

    x = [word_tokenize(d) for d in article_list]
    x = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in x]
    x = [d[:article_max_len] for d in x]
    x = [d + (article_max_len - len(d)) * [word_dict["<padding>"]] for d in x]
    
    if step == "valid":
        return x
    else:        
        y = [word_tokenize(d) for d in title_list]
        y = [[word_dict.get(w, word_dict["<unk>"]) for w in d] for d in y]
        y = [d[:(summary_max_len - 1)] for d in y]
        return x, y


def batch_iter(inputs, outputs, batch_size, num_epochs):
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    num_batches_per_epoch = (len(inputs) - 1) // batch_size + 1
    for epoch in range(num_epochs):
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(inputs))
            yield inputs[start_index:end_index], outputs[start_index:end_index]


def get_init_embedding(reversed_dict, embedding_size):
    glove_file = "glove/glove.6B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)

    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32)

        word_vec_list.append(word_vec)

    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)

def get_init_embedding_feats(word_dict , reversed_dict, embedding_size):
   
    print("Loading Lists...")
    train_article_list = get_text_list(train_article_path, False)
    train_title_list = get_text_list(train_title_path, False)

    print("Loading TF-IDF...")
    tf_idf_list = tf_idf_generate(train_article_list+train_title_list)
    
    print("Loading Pos Tags...")
    pos_list , postags_for_named_entity = get_pos_tags_dict(word_dict.keys())

    print("Loading Named Entity...")
    named_entity_recs = named_entity(postags_for_named_entity) 
    
    print("Loading Glove vectors...")

    glove_file = "glove/glove.6B.300d.txt"
    word2vec_file = get_tmpfile("word2vec_format.vec")
    glove2word2vec(glove_file, word2vec_file)
    print("Loading Glove vectors...")
    word_vectors = KeyedVectors.load_word2vec_format(word2vec_file)
     
    
    used_words = 0
    word_vec_list = list()
    for _, word in sorted(reversed_dict.items()):
        try:
            word_vec = word_vectors.word_vec(word)
            if word in tf_idf_list:
              v= tf_idf_list[word]
              rich_feature_array = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array)
            else:
              v=0
              rich_feature_array = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array)

            if word in pos_list:
              v=pos_list[word]
              rich_feature_array_2 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_2)
            else:
              v=0
              rich_feature_array_2 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_2) 

            if word in named_entity_recs:
              v=named_entity_recs[word]
              rich_feature_array_3 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_3)
            else:
              v=0
              rich_feature_array_3 = np.array([v,v,v,v,v,v,v,v,v,v])
              word_vec = np.append(word_vec, rich_feature_array_3)  
          
            used_words += 1
        except KeyError:
            word_vec = np.zeros([embedding_size], dtype=np.float32) #to generate for <padding> and <unk>
        
        
        word_vec_list.append(np.array(word_vec))

    print("words found in glove percentage = " + str((used_words/len(word_vec_list))*100) )
          
    # Assign random vector to <s>, </s> token
    word_vec_list[2] = np.random.normal(0, 1, embedding_size)
    word_vec_list[3] = np.random.normal(0, 1, embedding_size)

    return np.array(word_vec_list)

    ''' Functions for adding embedding POS and NER, TF*IDF'''
    def tf_idf_generate(sentences):
   
        # our corpus
        data = sentences

        cv = CountVectorizer()

        # convert text data into term-frequency matrix
        data = cv.fit_transform(data)

        tfidf_transformer = TfidfTransformer()

        # convert term-frequency matrix into tf-idf
        tfidf_matrix = tfidf_transformer.fit_transform(data)

        # create dictionary to find a tfidf word each word
        word2tfidf = dict(zip(cv.get_feature_names(), tfidf_transformer.idf_))

        return word2tfidf



#ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'

def get_pos_tags_dict(words):
    #sent = nltk.word_tokenize(sent)
    #print(sent)
    post_tags_for_words = nltk.pos_tag(words)

    pos_list ={}
    #sent = preprocess(ex)
    for word,pos in post_tags_for_words:
        pos_list[word] = pos
    #print(pos_list)

    import pandas as pd
    df = pd.DataFrame(list(pos_list.items()))
    df.columns = ['word', 'pos']
    df.pos = pd.Categorical(df.pos)
    df['code'] = df.pos.cat.codes
    #print(df)

    pos_list ={}
    for index, row in df.iterrows():
       pos_list[row['word']] = row['code']
    print(pos_list)
    return pos_list , post_tags_for_words


#https://nlpforhackers.io/named-entity-extraction/
def named_entity(post_tags_for_words):
    names = ne_chunk(post_tags_for_words)
    names_dict = {}
    for n in names:
        if (len(n) == 1):
            named_entity = str(n).split(' ')[0][1:]
            word = str(n).split(' ')[1].split('/')[0]
            names_dict[word] = named_entity
    print (names_dict)

    import pandas as pd
    df = pd.DataFrame(list(names_dict.items()))
    df.columns = ['word', 'pos']
    df.pos = pd.Categorical(df.pos)
    df['code'] = df.pos.cat.codes
    #print(df)

    names_dict ={}
    for index, row in df.iterrows():
        names_dict[row['word']] = row['code']
    print(names_dict)
    return names_dict

