import re
import openpyxl
import itertools

import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    data_tmp = pd.read_excel('data/accident.xlsx')
    # dtmp = data_tmp[data_tmp.columns[1:]]
    data_tmp['공종세부'] = data_tmp.공종.map(lambda x: str(x).split('>')[-1].strip())
    data_tmp['사고객체세부'] = data_tmp.사고객체.map(lambda x: str(x).split('>')[-1].strip())
    data_tmp['부위세부'] = data_tmp.부위.map(lambda x: str(x).split('/')[-1].strip())
    
    col = ['사고명', '인적사고', '공종세부', '사고객체세부', '작업프로세스', 
           '부위세부', '사고경위', '사고원인', '구체적 사고원인']
    dtmp = data_tmp[col]
    data_tmp['doc'] = [' '.join(map(str, dtmp.iloc[i])) for i in range(len(dtmp))]
    
    return data_tmp


stopwords = ['[a-zA-Z]*?', '날씨', '기온', '습도', '\(.*?\)', '/', 
            '\\d{1,5}-\\d{1,5}[생][활][권]',
            '\\d{1,5}[블][럭]',
            '\\d{1,5}[동]',
            '\\d{1,5}[번][지]',
            '\\d{1,5}[지]',
            '\\d{1,5}[층]',
            '\\d{1,3}[호][선]',
            '\\d{1,2}[월]',
            '\\d{1,2}[일]',
            '\\d{1,5}[%]', 
            '\\d{1,3}[℃]', 
            '\\d{1,5}[~]\\d{1,5}[개]', 
            '\\d{1,5}[~]\\d{1,5}[%]', 
            '\\d{2,4}-\\d{1,2}-\\d{1,2}', 
            '\\d{1,5}-\\d{1,5}', 
            '\\d{2,4}[년]\s\\d{1,2}[월]\s\\d{1,2}[일]', 
            '\\d{2,4}[년]\s\\d{1,2}[월]\\d{1,2}[일]', 
            '\\d{2,4}.\\d{1,2}.\\d{1,2}.', 
            '\\d{2,4}[.]\\d{1,2}[.]\\d{1,2}[.]', 
            '\\d{2,4}[.]\s\\d{1,2}[.]\s\\d{1,2}[.]', 
            '\\d{1,2}[시]\s\\d{1,2}[분][경]', 
            '\\d{1,2}[시]\\d{1,2}[분][경]', 
            '\\d{1,2}[시]\\d{1,2}[분]', 
            '\\d{1,3}[:]\\d{1,3}[분][경]',
            '\\d{1,3}[:]\\d{1,3}', 
            '\\d{1,5}~\\d{1,5}명', 
            '\\d{1,5}[~]\\d{1,5}[인]', 
            '\\d{1,5}[명]', 
            '\\d{1,5}[인]', 
            '\\d{1,5}[~]\\d{1,5}층',
            # '\\D[*]',
            # '[*]\\D',
            '[*][가-힣]',
            '[가-힣][*]',
            # '\\D{1,2}[*]\\D{1,2}',
            '\\d{1,3}[,]', '\\d{1,4}[만][원]', '\\d{1,4}[억][원]', '\\d{1,4}[억]', 
            '0명', '미만', '이상', '내국인', '외국인', '건축물', '건축',
            '안전방호', '개인보호', '피해없음', '해당없음',
            '정규작업', '대상현장', '사고발생', '공사현장',
            '기타', '없음',
            '\\d{1,5}\\w{1,5}',
            '\\w{1,10}[병][원]', '\\w{1,10}[교][회]', '\\w{1,10}[학][교]',
            '\\w{1,10}[빌][딩]', '\\w{1,10}[건][설]',
            '\\w{1,10}[군]\s\\w{1,10}[면]\s\\w{1,10}[리]',
            '\\w{1,10}[도|시|구|읍]\s\\w{1,10}[시|군|구|읍|면|동|리]',
            '[*]', '-', '~', ':', '>', 'ㆍ', ',', '\.', '`']


def cleaning(string):
    stopwords_ = '|'.join(stopwords)
    
    s_ = re.sub(stopwords_, '', string)
    s_ = re.sub('\n', ' ', s_)
    s_ = re.sub(' +', ' ', s_).strip()
    
    return s_


def cosim_mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
    
    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = cosine_similarity(candidate_embeddings, doc_embedding)

    # 각 키워드들 간의 유사도
    word_similarity = cosine_similarity(candidate_embeddings)

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # keywords_idx = [2]
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    # 만약, 2번 문서가 가장 유사도가 높았다면
    # ==> candidates_idx = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10 ... 중략 ...]
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    # ex) top_n = 5라면, 아래의 loop는 4번 반복됨.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)

    return [words[idx] for idx in keywords_idx], [word_doc_similarity[idx][0] for idx in keywords_idx]


def cor_mmr(doc_embedding, candidate_embeddings, words, top_n, diversity):
    # 문서와 각 키워드들 간의 유사도가 적혀있는 리스트
    word_doc_similarity = np.array(list(map(lambda x: pearsonr(x, doc_embedding[0])[0],  candidate_embeddings)))[:, np.newaxis]

    # 각 키워드들 간의 유사도
    word_similarity = pd.DataFrame(candidate_embeddings).T.corr().to_numpy()

    # 문서와 가장 높은 유사도를 가진 키워드의 인덱스를 추출.
    keywords_idx = [np.argmax(word_doc_similarity)]

    # 가장 높은 유사도를 가진 키워드의 인덱스를 제외한 문서의 인덱스들
    candidates_idx = [i for i in range(len(words)) if i != keywords_idx[0]]

    # 최고의 키워드는 이미 추출했으므로 top_n-1번만큼 아래를 반복.
    for _ in range(top_n - 1):
        candidate_similarities = word_doc_similarity[candidates_idx, :]
        target_similarities = np.max(word_similarity[candidates_idx][:, keywords_idx], axis=1)

        # MMR을 계산
        mmr = (1-diversity) * candidate_similarities - diversity * target_similarities.reshape(-1, 1)
        mmr_idx = candidates_idx[np.argmax(mmr)]

        # keywords & candidates를 업데이트
        keywords_idx.append(mmr_idx)

    return [words[idx] for idx in keywords_idx], [word_doc_similarity[idx][0] for idx in keywords_idx]


def is_float(num):
    try:
        float(num)
        return True
    except:
        return False


def josa_delete(string):
    if string[-4:] == '되었다고' or string[-4:] == '않았으며' or \
        string[-4:] == '확인하고' or string[-4:] == '뛰었으나':
        return string[:-4].strip()
    
    elif string[-3:] == '있었음' or string[-3:] == '이었음' or \
        string[-3:] == '했으나' or string[-3:] == '하였음' or \
        string[-3:] == '잃으며' or string[-3:] == '되면서' or \
        string[-3:] == '하였고':
        return string[:-3].strip()
    
    elif string[-2:] == '으로' or string[-2:] == '이하' or string[-2:] == '에서' or \
        string[-2:] == '에도' or string[-2:] == '되어' or string[-2:] == '하여' or \
        string[-2:] == '하는' or string[-2:] == '보다' or string[-2:] == '하러' or \
        string[-2:] == '에게' or string[-2:] == '확인' or string[-2:] == '올라' or \
        string[-2:] == '하지' or string[-2:] == '인한' or string[-2:] == '올라' or \
        string[-2:] == '무엇' or string[-2:] == '잃고' or string[-2:] == '잃어' or \
        string[-2:] == '있던' or string[-2:] == '하다'  or string[-2:] == '오다' or \
        string[-2:] == '하고' or string[-2:] == '하던':
        return string[:-2].strip()
    
    elif string[-1] == '은' or string[-1] == '는' or string[-1] == '을' or \
        string[-1] == '들' or string[-1] == '를' or string[-1] == '이' or \
        string[-1] == '가' or string[-1] == '의' or string[-1] == '에' or \
        string[-1] == '과' or string[-1] == '과' or string[-1] == '로' or \
        string[-1] == '으' or string[-1] == '임' or string[-1] == '한' or \
        string[-1] == '해' or string[-1] == '함' or string[-1] == '중' or \
        string[-1] == '내' or string[-1] == '된' or string[-1] == '되': # or string[-1] == '인'
        return string[:-1].strip()
    
    elif is_float(string):
        return ''
    
    # elif string[:2] == '인한' or string[-2:] == '올라':
    #     return string[2:].strip()
    
    else:
        return string
