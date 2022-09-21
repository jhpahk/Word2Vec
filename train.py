import os
import time

from gensim.models import Word2Vec
from data import *


# 학습에 사용 가능한 데이터 파악
start_time = time.time()
print("학습 데이터 목록 구성 시작")

data_dir = 'C:/Users/USER/Desktop/Quiz_Project/kor_Word2Vec/Word2Vec/data'
file_list = os.listdir(data_dir)

print("학습 데이터 목록 구성 완료")
print(f"-> {file_list}")
end_time = time.time()
print(f"소요 시간: {int((end_time - start_time) // 60)}min {(end_time - start_time) % 60:.2f}sec\n")


# 학습 데이터 불러오기 및 전처리
start_time = time.time()
print("학습 데이터 불러오기 및 전처리 시작")

raw_data = []
for filename in file_list:
    raw_data += load_data("data/" + filename)
data = process_data(raw_data)

print("학습 데이터 불러오기 및 전처리 완료")
end_time = time.time()
print(f"소요 시간: {int((end_time - start_time) // 60)}min {(end_time - start_time) % 60:.2f}sec\n")


# 학습 시작 / parameters: (sg -> 0이면 CBOW, 1이면 SkipGram | hs = 0 & negative > 0 -> negative sampling 적용)
start_time = time.time()
print("학습 시작")

model = Word2Vec(sentences=data,
                 sg=1, vector_size=300, window=5, hs=0, negative=10, min_count=4, workers=8)

print("학습 완료")
end_time = time.time()
print(f"소요 시간: {int((end_time - start_time) // 60)}min {(end_time - start_time) % 60:.2f}sec\n")


# 결과 저장
curtime = time.strftime("%Y%m%d_%H%M%S")
result_filename = f"word2vec_kor_{curtime}.model"
model.save(result_filename)


# 결과 확인
print(f"말뭉치 개수 -> {model.corpus_count}")
print(f"말뭉치 내 전체 단어수 -> {model.corpus_total_words}")
