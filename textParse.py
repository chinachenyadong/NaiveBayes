#!/usr/bin/env python

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*', bigString)
    return [ tok.lower() for tok in listOfTokens if len(tok) > 2 ]

def spamTest():
    for i in range(1,26):
        wordList = textParse( open('./email/ham/%d.txt' % i).read() )
        #wordList = textParse( open('/Users/chenyadong/Program/c++/ml/nb/email/ham/%d.txt' % i).read() )
        fp = open( './email/hamParse/%d.dat' % i , 'w' )

        for item in wordList:
            fp.write(item + ' ')
        wordList = textParse( open('./email/spam/%d.txt' % i).read() )
        fp = open( './email/spamParse/%d.dat' % i , 'w' )
        for item in wordList:
            fp.write(item + ' ')
    
spamTest()
