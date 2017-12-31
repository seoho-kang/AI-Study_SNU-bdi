# -*- coding: utf-8 -*-
# @author: sana

import nltk, re

### file opening options ###

# 1) 로컬 파일 열기
f = open('BROWN1_A1.txt', 'rU')
# 데이터 읽기
data = f.read()

# 2) url 통해 웹 파일 열기
# http://www.nltk.org/book/ch03.html 참조
from urllib import request
url = 'http://www.gutenberg.org/files/2554/2554.txt'
f = request.urlopen(url)
# 데이터 읽기
data = f.read()

# 3) nltk corpus 열기
# http://www.nltk.org/book/ch02.html 참조
from nltk.corpus import gutenberg
data = gutenberg.raw('austen-emma.txt')


# tokenize
words = []
words = data.split()	# words = nltk.word_tokenize(data)

# 정규표현식
wh = [word for word in words if re.search(r'^(who|which|when|what|where|why|whose|whom)$', word.lower())]

# 출력
for w in wh:
	print w 	# print(w)

