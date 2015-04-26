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

class NaiveBayes
{
	private:
		vector< vector<string> > list_of_docs;  //词条向量
		vector<int> list_classes;
		map<string, int> my_vocab_list; //单词列表
		int *return_vec;
		vector< vector<int> > train_mat;
		vector<float> p0vect;
		vector<float> p1vect;
		float p_abusive;
		ifstream fin;
		ofstream fout;
		int test_data_num;

	public:
		NaiveBayes()
		{
			cout << "please input the num of test data which should be less than 24 : " << endl;
			cin >> test_data_num;
			vector<string> vec;
			string word;
			string filename;
			char buf[3];
			string buf_str;

			for (int i = test_data_num + 1; i <= 25; ++i)
			{
				sprintf(buf, "%d", i);
				vec.clear();
				buf_str = buf;
				filename = "./email/hamParse/" + buf_str + ".dat";
				fin.open(filename.c_str());
				if (!fin)
				{
					cerr << "open the file " << filename << " error" << endl;
					exit(1);
				}
				while (fin >> word)
				{
					vec.push_back(word);
				}
				list_of_docs.push_back(vec);
				list_classes.push_back(0);
				filename.clear();
				fin.close();
			}

			for (int i = test_data_num + 1; i <= 25; ++i)
			{
				sprintf(buf, "%d", i);
				vec.clear();
				buf_str = buf;
				filename = "./email/spamParse/" + buf_str + ".dat";
				fin.open(filename.c_str());
				if (!fin)
				{
					cerr << "open the file " << filename << " error" << endl;
				}
				while (fin >> word)
				{
					vec.push_back(word);
				}
				list_of_docs.push_back(vec);
				list_classes.push_back(1);
				filename.clear();
				fin.close();
			}
		}

		~NaiveBayes()
		{
			fin.close();
			fout.close();
			list_of_docs.clear();
			list_classes.clear();
			my_vocab_list.clear();
			train_mat.clear();
			p0vect.clear();
			p1vect.clear();
			
		}

		void create_vocab_list()
		{
			vector< vector<string> >::iterator it = list_of_docs.begin();
			int index = 1; // the location of the vocabulary
			while ( it != list_of_docs.end() )
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
		void bag_of_words_to_vec(int idx)
		{
			int len = my_vocab_list.size() + 1;
			return_vec = new int[len]; //区别new int[len];()为0  	
			fill(return_vec, return_vec+len, 0);
			// 第idx个邮件的bag转为vec
			vector< vector<string> >::iterator it = list_of_docs.begin() + idx - 1;
			vector<string> vec = *it;
			vector<string>::iterator itt = vec.begin();

			int pos = 0;
			while ( itt != vec.end() )
			{
				pos = my_vocab_list[*itt];
				if (pos != 0)
				{
					return_vec[pos] += 1;
				}
				++itt;
			}
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
			
			for (unsigned int i = 1; i <= list_of_docs.size(); ++i)
			{
				bag_of_words_to_vec(i);
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
			
			cout << endl;
		}

		int classify_NB(const char *filename)
		{
			return_vec = new int[ my_vocab_list.size() + 1 ]();
			
			fin.open(filename);
			if (!fin)
			{
				cerr << "fail to open the file" << filename << endl;
				exit(1);
			}
			string word;
			while (fin >> word)
			{
				int pos = my_vocab_list[word];
				if (pos != 0)
				{
					return_vec[pos] += 1;
				}
			}
			fin.close();

			cout << endl;
			float p1 = inner_product(p1vect.begin()+1, p1vect.end(), return_vec+1, 0) + log(p_abusive);
			float p0 = inner_product(p0vect.begin()+1, p0vect.end(), return_vec+1, 0) + log(1-p_abusive);

			cout << "p1 = " << p1 << "  p0 = " << p0 << endl;

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

		void get_error_rate()
		{
			string filename;
			char buf[3];
			string buf_str;
			int error_count = 0;
			for (int i = 1; i <= test_data_num; ++i)
			{
				sprintf(buf, "%d", i);
				buf_str = buf;
				filename = "./email/hamParse/" + buf_str + ".dat";
				if ( classify_NB( filename.c_str() ) != 0 )
				{
					++error_count;
				}

				filename = "./email/spamParse/" + buf_str + ".dat";
				if ( classify_NB( filename.c_str() ) != 1 )
				{
					++error_count;
				}
			}

			cout << "the error rate is : " << (float)error_count / (2*test_data_num) << endl;
		}
};

int main()
{
	NaiveBayes nb;
	nb.create_vocab_list();
	nb.get_train_matrix();
//	nb.print();
	nb.train_NB0();

	char doc1_to_classify[] = "./email/hamParse/1.dat";
	char doc2_to_classify[] = "./email/spamParse/1.dat";
	cout << "doc1 classified as : " << nb.classify_NB( doc1_to_classify ) << endl;
	cout << "doc2 classified as : " << nb.classify_NB( doc2_to_classify ) << endl;
	
	nb.get_error_rate();
	return 0;
}
