/*
 * 1. transfer func from docs list to vocabulary list
 * 2. training function on Navie Bayes Classifier
 * 3. naive bayes classify function
 */

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <numeric>

using namespace std;

string posting_list[6][10] = {
	{"my","dog","has","flea","problems","help","please","null"},  
	{"maybe","not","take","him","to","dog","park","stupid","null"},  
	{"my","dalmation","is","so","cute","I","love","him","null"},  
	{"stop","posting","stupid","worthless","garbage","null"},  
	{"mr","licks","ate","my","steak","how","to","stop","him","null"},  
	{"quit","buying","worthless","dog","food","stupid","null"}  
};  
int class_vec[6] = {0,1,0,1,0,1};

class NaiveBayes
{
	private:
		vector< vector<string> > list_of_posts;  //词条向量
		vector<int> list_classes;
		map<string, int> my_vocab_list; //单词列表
		int *return_vec;
		vector< vector<int> > train_mat;
		vector<float> p0vect;
		vector<float> p1vect;
		float p_abusive;

	public:
		NaiveBayes()
		{
			// posting_list --> list_of_posts
			vector<string> vec;
			for (int i = 0; i < 6; ++i)
			{
				vec.clear();
				for (int j = 0; posting_list[i][j] != "null"; ++j)
				{
					vec.push_back( posting_list[i][j] );
				}
				list_of_posts.push_back( vec );
			}

			// class_vec --> list_classes
			for (int i = 0; i < (int)(sizeof(class_vec) / sizeof(class_vec[0])); ++i)
			{
				list_classes.push_back( class_vec[i] );
			}
		}

		void create_vocab_list()
		{
			vector< vector<string> >::iterator it = list_of_posts.begin();
			int index = 1; // the location of the vocabulary
			while ( it != list_of_posts.end() )
			{
				vector<string> vec = *it;
				vector<string>::iterator tmp_it = vec.begin();

				while ( tmp_it != vec.end() )
				{
					if ( my_vocab_list[*tmp_it] == 0 )
					{
						my_vocab_list[*tmp_it] = index++;
					}
					++tmp_it;
				}
				++it;
			}	
		}

		// set some one doc to vec with 0 and 1.
		void set_of_words_to_vec(int idx)
		{
			cout << "set of words to vec begin the document id is : " << idx << endl;
			int len = my_vocab_list.size() + 1;
			return_vec = new int[len]; //区别new int[len];()为0  	
			fill(return_vec, return_vec+len, 0);
			for (int i = 0; posting_list[idx][i] != "null"; ++i)
			{
				int pos = my_vocab_list[ posting_list[idx][i] ];
				if ( pos != 0 )
				{
					return_vec[pos] = 1;
				}
			}
			cout << endl;
		}

		void print()
		{
			cout << "print the train matrix begin :" << endl;	
			vector< vector<int> >::iterator it = train_mat.begin();
			while (it != train_mat.end())
			{
				vector<int> vec = *it;
				vector<int>::iterator itt = vec.begin();
				while (itt != vec.end())
				{
					cout << *itt << " ";
					++itt;
				}
				cout << endl;
				++it;
			}
		}

		void get_train_matrix()
		{
			cout << "get train matrix begin : " << endl;
			train_mat.clear();
			
			for (int i = 0; i < 6; ++i)
			{
				set_of_words_to_vec(i);
				vector<int> vec( return_vec, return_vec + my_vocab_list.size() + 1 );
				train_mat.push_back(vec);
				delete[] return_vec;
			}
		}

		void train_NB0()
		{
			int num_train_docs = train_mat.size();
			cout << "num_train_docs = " << num_train_docs << endl;
			// calculate the sum of the abusive classes
			int sum = accumulate(list_classes.begin(), list_classes.end(), 0);
			cout << "sum = " << sum << endl;	
			p_abusive = (float) sum / num_train_docs;
			cout << "p_abusive = " << p_abusive << endl;

			//  p(w|c0)
			//  初始取1，防止乘积为0。和取2
			p0vect.resize(train_mat[0].size(), 1);
			//  p(w|c1)
			p1vect.resize(train_mat[0].size(), 1);
			printf("p0num.size() = %d , p1num.size() = %d\n", (int)p0vect.size(), (int)p1vect.size());	

			// the total number of words in non-abusive docs
			float p0Denom = 2.0;
			// the total number of words in abusive docs
			float p1Denom = 2.0;

			// calculate the p0num, p1num, p0Denom, p1Denom
			for (unsigned long i = 0; i < list_classes.size(); ++i)
			{
				if (list_classes[i] == 1)
				{
					for (unsigned long j = 0; j < p1vect.size(); ++j)
					{
						p1vect[j] += train_mat[i][j];
						if (train_mat[i][j] == 1)
							++p1Denom;
					}
				}
				else
				{
					for (unsigned long j = 0; j < p0vect.size(); ++j)
					{
						p0vect[j] += train_mat[i][j];
						if (train_mat[i][j] == 1)
							++p0Denom;
					}
				}
			}

			for (unsigned long i = 0; i < p1vect.size(); ++i)
			{
				p0vect[i] = log( p0vect[i] / p0Denom );
				p1vect[i] = log( p1vect[i] / p1Denom );
			}
			
			cout << "print the p0vect values : ";
			for (unsigned long i = 0; i < p0vect.size(); ++i)
				cout << p0vect[i] << " ";
			cout << "\nprint the p1vect values : ";
			for (unsigned long i = 0; i < p1vect.size(); ++i)
				cout << p1vect[i] << " ";
			cout << endl;
		}

		int classify_NB(string *doc_to_classify)
		{
			return_vec = new int[ my_vocab_list.size() + 1 ]();
			for (int i = 0; doc_to_classify[i] != "null"; ++i)
			{
				int pos = my_vocab_list[ doc_to_classify[i] ];
				if (pos != 0)
				{
					return_vec[pos] = 1;
				}
			}

			for (unsigned long i =0; i < my_vocab_list.size() + 1; ++i)
				cout << return_vec[i] << " ";
			cout << endl;
			float p1 = inner_product(p1vect.begin()+1, p1vect.end(), return_vec+1, 0) + log(p_abusive);
			float p0 = inner_product(p0vect.begin()+1, p0vect.end(), return_vec+1, 0) + log(1-p_abusive);

			cout << "p1 = " << p1 << endl;
			cout << "p0 = " << p0 << endl;

			if (p1 > p0)
			{
				return 1;
			}
			else
			{
				return 0;
			}
			return 0;
		}
};

int main()
{
	NaiveBayes nb;
	nb.create_vocab_list();
	nb.get_train_matrix();
	nb.print();
	nb.train_NB0();

	string doc1_to_classify[] = {"love", "my", "dalmation", "null"};
	string doc2_to_classify[] = {"stupid", "garbage", "null"};
	cout << "doc1 classified as : " << nb.classify_NB( doc1_to_classify ) << endl;
	cout << "doc2 classified as : " << nb.classify_NB( doc2_to_classify ) << endl;
	
	return 0;
}
