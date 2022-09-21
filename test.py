import matplotlib.pyplot as plt
from gensim.models import word2vec


plt.rc('font', family='Malgun Gothic')


# 바 그래프 그리기
def showGraph(bargraph):

     xtick = [item[0] for item in bargraph] # 단어
     ytick = [item[1] for item in bargraph] # 유사도
     
     plt.figure()
     plt.bar(xtick, ytick,
             color=['#06c2ac', '#c79fef', '#ff796c', '#aaff32', '#0485d1', '#d648d7', '#a5a502', '#d8dcd6', '#5ca904', '#fffe7a'])


# 모델 불러오기
model_filename = 'word2vec_kor_20220922_031620.model'
model = word2vec.Word2Vec.load(("saves/" + model_filename))
print(f"{model_filename} 불러오기 완료\n")


# 유사도 구하기
# 입력받은 단어와 유사도가 높은 단어 10개를 리스트로 반환
while True:
        target = input("단어를 입력하세요: ")
        if not target:
                break
        
        bargraph = model.wv.most_similar(positive=[target], topn=10)
        showGraph(bargraph)

        plt.show()