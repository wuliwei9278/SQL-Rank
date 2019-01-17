# SQL-Rank: A Listwise Approach to Collaborative Ranking
## Announcement:
- The paper has been accepted for oral presentation (24.9% acceptance rate) by ICML’18, Stockholm, Sweden (https://icml.cc/Conferences/2018/AcceptedPapersInitial). One can find the paper on my personal website: http://anson.ucdavis.edu/~liweiwu/papers/icml18.pdf.
- I gave an oral presentation about this work at Stockholm, Sweden on July, 2018.
- You can cite the work using the BibTex below:

```
@inproceedings{wu2018sql,
  title={SQL-Rank: A Listwise Approach to Collaborative Ranking},
  author={Wu, Liwei and Hsieh, Cho-Jui and Sharpnack, James},
  booktitle={Proceedings of Machine Learning Research (35th International Conference on Machine Learning)},
  volume={80},
  year={2018}
}
```

## Description: 
This repo consists of sample train/test data and SQL-Rank algorithm julia source code. It only works on Julia 0.6 so it requires modification on Julia 1.0. 



## Instructions on how to run the code:
Our trained model can be tested in terms of Precision@1, 5, 10 and objective function. The codes are specifically optimized for implicit feedback dataset only, therefore the ratings in sample datasets are always 1.



1. Prepare a dataset of the form (user, item, ratings) triple in csv file. (Example: ml1m_oc_50_train_ratings.csv)



2. Use command line to go to the repo folder and start Julia 

	```
	$ julia
	```


3. Type the following in Julia command line to load the functions for SQL-Rank algorithm:
	```
	julia> include("sqlrank12.jl")
	```
	


5. Type in Julia command line 
	```
	julia> main("ml1m_oc_50_train_ratings.csv", "ml1m_oc_50_test_ratings.csv", 0.2, 0.9, 1, 4, 100, 3)
	```
  	 Use `ctrl - c` to stop the program after it starts printing the results for the 1st iteration and 	type again 
	```
	julia> main("ml1m_oc_50_train_ratings.csv", "ml1m_oc_50_test_ratings.csv", 0.2, 0.9, 1, 4, 100, 3)
	```
	, where first two arguments "ml1m_oc_50_train_ratings.csv", "ml1m_oc_50_test_ratings.csv" are training & test data file paths; 0.2 is the initial learning rate; 0.9 and 1 means multiply the initial learning rate by 0.9 every 1 iteration; 4, 100 and 3 are the regularization parameter lambda, rank r and negative sampling ratio tho in the model.
	
	One can replace the arguments for the main function to tune the model. 

	The reason to type the same command twice in Julia command line is that the first time Julia will first compile the codes and the second time the codes will run much faster because the compilation time is saved.
