target:
	./textParse.py
	g++ NB_4.cpp
	./a.out

clean:
	rm ./email/spamParse/* ./email/hamParse/* a.out
