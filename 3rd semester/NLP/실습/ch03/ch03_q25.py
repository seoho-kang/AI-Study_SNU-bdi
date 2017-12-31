# -*- coding: utf-8 -*-
# @author: sana

import nltk, re

# Pig Latin is a simple transformation of English text. Each word of the text is converted as follows: move any consonant (or consonant cluster) that appears at the start of the word to the end, then append ay, e.g. string → ingstray, idle → idleay.  http://en.wikipedia.org/wiki/Pig_Latin


def toPigLatin(word):
	# qu 또는 자음으로 시작하는 문자열 
	# \W: 비-단어 문자 = [^A-Za-z0-9_]
    r = r'(qu|[^[aeiouy|\W]*)(.*)' # consider y as vowel
    tup = re.findall(r, word, re.IGNORECASE)[0]

    print tup
    # tup[0] = 정규표현식에 해당하는 부분
    # tup[1] = 나머지 부분
    
    return tup[1] + tup[0] + 'ay'


tokens = ['string', 'idle', 'PigLatin', 'quiet', 'style', 'yellow']
for token in tokens:
	print toPigLatin(token)