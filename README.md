# NLP with Quora Questions
## Overview
In this repository I explore different methods of Natural Language Processing on the Quora Questions database. The goal is to determine if two questions are identical. This is an important feature for a business like Quora where they have millions of users: if they can accurately determine if questions are similar then they can reduce their overhead by using a cached answer instead of accessing a database.

## Table of Contents
  1. [**EDA**](https://github.com/Graham-Broughton/NLP-Quora/edit/master/README.md#EDA)
  2. [**Feature Engineering**](https://github.com/Graham-Broughton/NLP-Quora/edit/master/README.md#FeatureEngineering)
  3. [**Baseline Model**](https://github.com/Graham-Broughton/NLP-Quora/edit/master/README.md#Logistic-Regression)
  4. [**XGBoost**](https://github.com/Graham-Broughton/NLP-Quora/edit/master/README.md#XGBoost)

## EDA
For my exploratory data analysis I analyzed the target distribution, various features of the questions and target distribution among engineered question features. I discovered a slight class imbalance, shown below, and proceeded with baseline models without any correction. 
![target distribution pie chart](src/images/Quora_dupes.png)
Out of the 404290 question pairs there were 537933 unique questions, 111780 repeated questions with a maximum repeat of 157. When seperated by target, there were more repeated questions in the duplicate class than there should have been according to the class distribution (44% instead of 37%).
![Repeated Questions Bar Chart](src/images/Quora_unique.png)
The log-scaled histogram of repeated questions, seen below, shows the vast majority of questions are either not repeated, or repeated a few times, with an extremely long tail.
![Repeated Questions Log-Histo](src/images/Quora_freq.png)
There were also three NaN values which I chose to impute with an empty string. This way, if Quora does not have a safeguard against empty queries, the model will be able to handle them.

## Feature Engineering
I started off with basic engineering based on the characteristics of the questions, then moved towards advanced tools including: the FuzzyWuzzy library, TF-idf Vectorizer and Truncated SVD, word vector embeddings and an assortment of distances between those embeddings. My basic engineering consisted of calculating: character length, difference of character length, absolute difference of character length, number of words, number of common words, ratio of common words, frequency of questions, frequency of both questions, absolute difference of frequencies and the sets of characters used. As seen in the violin and kernel plots, the number and ratio of common words will likely be an effective feature due to the difference of values between classes.
![Violin plots](src/images/Quora_violin.png)
![Kernel Plots](src/images/Quora_kernel.png)
FuzzyWuzzy is a python library for string matching which uses Levenshtein distance to compare the differences between the strings. Essentially it calculates the number of single character edits the two strings differ from one another. 
