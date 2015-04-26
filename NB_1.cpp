/*
 * transfer func from docs list to vocabulary list
 */

#include <iostream>
#include <map>
#include <set>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>

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
			for (int i = 0; i < sizeof(class_vec) / sizeof(class_vec[0]); ++i)
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

			map<string, int>::const_iterator itt = my_vocab_list.begin();
			while ( itt !=my_vocab_list.end() )
			{
				cout << itt->first << " " << itt->second << " ";
				++itt;
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
			cout << "print the return_vec begin :" << endl;	
			int len = my_vocab_list.size() + 1;
			cout << "len = " << len << endl;
			for (int i = 0; i < len; ++i)
			{
				cout << return_vec[i] << " ";
			}
			cout << endl;
			delete[] return_vec;
		}
};

int main()
{
	NaiveBayes nb;
	nb.create_vocab_list();
	nb.set_of_words_to_vec(5);
	nb.print();
	
	return 0;
}
