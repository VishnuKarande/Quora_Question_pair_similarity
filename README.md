 
# Quora Question Pairs

# Description

Over 100 million people visit Quora every month, so it's no surprise that many people ask similarly worded questions. Multiple questions with the same intent can cause seekers to spend more time finding the best answer to their question, and make writers feel they need to answer multiple versions of the same question. Quora values canonical questions because they provide a better experience to active seekers and writers, and offer more value to both of these groups in the long term. so main aim of project is that predicting whether pair of questions are similar or not. This could be useful to instantly provide answers to questions that have already been answered. Credits: Kaggle

# Problem Statement
Identify which questions asked on Quora are duplicates of questions that have already been asked.

# Sources/Useful Links
## Useful Links
Discussions : https://www.kaggle.com/anokas/data-analysis-xgboost-starter-0-35460-lb/comments

Kaggle Winning Solution and other approaches: https://www.dropbox.com/sh/93968nfnrzh8bp5/AACZdtsApc1QSTQc7X0H3QZ5a?dl=0

Blog 1 : https://engineering.quora.com/Semantic-Question-Matching-with-Deep-Learning

Blog 2 : https://towardsdatascience.com/identifying-duplicate-questions-on-quora-top-12-on-kaggle-4c1cf93f1c30

## Real world/Business Objectives and Constraints
The cost of a mis-classification can be very high.
You would want a probability of a pair of questions to be duplicates so that you can choose any threshold of choice.
No strict latency concerns.
Interpretability is partially important.

# Data Overview:
Train.csv contains 5 columns : qid1, qid2, question1, question2, is_duplicate. Total we have 404290 entries. Splitted data into train and test with 70% and 30%.

# Feature Extraction:
#### Basic Features - Extracted some features before cleaning of data as below.
1.freq_qid1 = Frequency of qid1's.

2.freq_qid2 = Frequency of qid2's.

3.q1len = Length of q1.

4.q2len = Length of q2.

5.q1_n_words = Number of words in Question 1.

6.q2_n_words = Number of words in Question 2.

7.word_Common = (Number of common unique words in Question 1 and Question 2).

8.word_Total =(Total num of words in Question 1 + Total num of words in Question 2).

9.word_share = (word_common)/(word_Total).

10.freq_q1+freq_q2 = sum total of frequency of qid1 and qid2.

11.freq_q1-freq_q2 = absolute difference of frequency of qid1 and qid2.

#### Advanced Features - Did some preprocessing of texts and extracted some other features. i am giving some definitions which are used below. Token- You get a token by splitting sentence by space , Stop_Word - stop words as per NLTK, Word -A token that is not a stop_word.

-cwc_min = common_word_count / (min(len(q1_words), len(q2_words))

-cwc_max = common_word_count / (max(len(q1_words), len(q2_words))

-csc_min = common_stop_count / (min(len(q1_stops), len(q2_stops))

-csc_max = common_stop_count / (max(len(q1_stops), len(q2_stops))

-ctc_min = common_token_count / (min(len(q1_tokens), len(q2_tokens))

-ctc_max = common_token_count / (max(len(q1_tokens), len(q2_tokens))

-last_word_eq = Check if Last word of both questions is equal or not (int(q1_tokens[-1] == q2_tokens[-1]))

-first_word_eq = Check if First word of both questions is equal or not (int(q1_tokens[0] == q2_tokens[0]))

-abs_len_diff = abs(len(q1_tokens) - len(q2_tokens))

-mean_len = (len(q1_tokens) + len(q2_tokens))/2

-fuzz_ratio = How much percentage these two strings are similar, measured with edit distance.

-fuzz_partial_ratio = if two strings are of noticeably different lengths, we are getting the score of the best matching lowest length substring.

-token_sort_ratio = sorting the tokens in string and then scoring fuzz_ratio.
-longest_substr_ratio = len(longest common substring) / (min(len(q1_tokens), len(q2_tokens))


## Features Extraction(text preprocessing techniques)
1-tfidf

2-Bow

3-AVG-W2V

# Demo:(q1,q2,q3,q4 are questions)

q1 = 'Where is the capital of India?'

q2 = 'What is the current capital of Pakistan?'

q3 = 'Which city serves as the capital of India?'

q4 = 'What is the business capital of India?'

(prediction)

rf.predict(query_point_creator(q1,q4))
### Ans
array([1], dtype=int64
(Not duplicate)

#### References:
https://www.kaggle.com/c/quora-question-pairs
https://www.kaggle.com/c/quora-question-pairs/discussion
https://github.com/seatgeek/fuzzywuzzy#usage , https://chairnerd.seatgeek.com/fuzzywuzzy-fuzzy-string-matching-in-python/
http://proceedings.mlr.press/v37/kusnerb15.pdf
https://github.com/CreatorGhost/Quora/blob/master/Quora%20Question%20Pair.ipynb.

