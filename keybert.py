import pandas as pd

from tqdm import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from soynlp.tokenizer import LTokenizer, MaxScoreTokenizer
from soynlp.noun import LRNounExtractor, LRNounExtractor_v2

from sentence_transformers import SentenceTransformer

from utils import (
    load_data,
    cleaning,
    cosim_mmr,
    cor_mmr,
    josa_delete
)


data_tmp = load_data()
data_tmp['clean_doc'] = data_tmp['doc'].map(lambda x: cleaning(x))

noun_extractor = LRNounExtractor_v2(verbose=False)
# noun_extractor = LRNounExtractor(verbose=False)
nouns = noun_extractor.train_extract(data_tmp.clean_doc)

candi_words = {} 

for word, r in nouns.items():
    if (r[0] <= 1000) and (len(word)>=3):
    #print('%8s:\t%.4f' % (word, r[0]))
        candi_words[word] = r[1]

tokenizer = MaxScoreTokenizer(scores=candi_words)
# tokenizer = LTokenizer(scores = candi_words)

model = SentenceTransformer("distiluse-base-multilingual-cased-v1")


top_n = 3
n_gram_range = (1,2)

result_ = []

if __name__ == '__main__':
    for i in tqdm(range(len(data_tmp))):
        keywords_ = {}
        
        keywords_['사고번호'] = data_tmp.사고번호[i]
        doc_ = data_tmp.clean_doc[i]
        
        soytokens = tokenizer.tokenize(doc_)
        soytokens = list(map(josa_delete, soytokens))

        count = CountVectorizer(ngram_range = n_gram_range).fit([' '.join(soytokens)])
        # count = CountVectorizer(ngram_range = n_gram_range).fit([doc_])
        candidates = count.get_feature_names_out()
        
        doc_embedding = model.encode([doc_])
        candi_embedding = model.encode(candidates)
        
        # distances = cosine_similarity(doc_embedding, candi_embedding)
        cosim_keywords, keyword_cosim = cosim_mmr(doc_embedding, candi_embedding, candidates, top_n=top_n, diversity=0.3)
        # cosim_keywords = [josa_delete(x) for x in cosim_keywords]
        
        keywords_.update({'cosim_keyword' + str(j+1): cosim_keywords[j] for j in range(len(cosim_keywords))})
        keywords_.update({'keyword' + str(j+1) + 'cosim': keyword_cosim[j] for j in range(len(keyword_cosim))})
        
        cor_keywords, keyword_cor = cor_mmr(doc_embedding, candi_embedding, candidates, top_n=top_n, diversity=0.3)
        # cor_keywords = [josa_delete(x) for x in cor_keywords]
        
        keywords_.update({'cor_keyword' + str(j+1): cor_keywords[j] for j in range(len(cor_keywords))})
        keywords_.update({'keyword' + str(j+1) + 'cor': keyword_cor[j] for j in range(len(keyword_cor))})
        
        result_.append(keywords_)


keywords = pd.DataFrame(result_)
# keywords.to_csv('result/대표키워드.csv', header = True, index = False)
keywords.to_csv('result/대표키워드.csv', header = True, index = False, encoding = 'utf-8-sig')
