*------- nlp_fianl_assignment toy example -------*


# file 구조
nlp_final_assignment(dir)
|
|___data(dir)
|     |
|     |___ news(dir)
|            |
|            |___여기에 news 기사 .txt 파일을 넣어주세요
|
|
|___parsing(copy).py(file)
|
|__word2vec(file)
 
 
# word2vec 파일 설명 
word2vec 파일은 size=100, iter=5(default), sg=0(default)(<- CBOW를 의미합니다) 로 학습된 것입니다.
 
 
# 실행
nlp_final_assignment에서 python parsing_copy.py로 실행해주세요