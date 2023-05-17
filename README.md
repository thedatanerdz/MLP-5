# MLP -5
- Sentiment Analysis classifier of amazon customer reviews


Industry 
Retail | Ecommerce

Skills
Python | Sentiment Analysis | Huggingface Roberta Transformers  | NLP | NLTK | Transformers Pipeline
| VADER | EDA  | Data visualization | Data cleaning | Data manipulation 

Problem statements
Classify amazon reviews to deliver better customer service.

Data Description


Data Structure 
The important columns in the dataset used was 
Id
Product Id
Helpfulness Numerator
HelpfulnessDenominator
Summary
Score
Text
Methods
Use different models and test which one is more accurate

Quick EDA
Get shape of dataframe
Extract 500 reviews 
Count star rating in bar graph 

Basic NLTK
Split sentence up with tokenizers 
Part of speech tagging (use library docs for abbreviation definitions)
Put tagged parts of speech into entities using chuncker (group them into chunks of texts)
Pretty print output 

VADER Sentiment Scoring
FUNCTION: Analysis groups of words and groups them into negative, neutral or positive categories. Looks at each word individually and scores each word individually.  
CON: Doesn’t account for relationships between words 
OMITS: Stop words with not neg/neu/pos connotation .ie and, the 
tdqm tracks progress with loops 
Make sentiment analyser  object used 
Test using own input ‘I am so happy to get neg/neu/pos scores 
We test polarity score on a small section of the dataset called example
Then we run it on the entire dataset using a for loop 
Store results in dic called res 
Store dictionary  in dataframe with transpose to flip dataframe 
Reset index to merge onto original dataframe (left merge)
We have sentiment score and metadata 

Roberta Pretrained Model
TRANSFORMER BASED DEEP LEARNING MODELS CAN PICK UP
Softmax smoothes out between 0 and 1 (normalize the data)

In Python, the softmax function is a mathematical function that is often used in machine learning to convert a vector of real numbers into a probability distribution that sums up to 1.

The softmax function takes as input a vector of real numbers and applies a mathematical formula to each element in the vector. 

The resulting output vector from the softmax function can be interpreted as a probability distribution over the possible categories, with each element in the output vector representing the probability of the corresponding category.

The softmax function is often used in the output layer of neural networks for classification problems where the goal is to predict the probability distribution of the target class for a given input. Softmax function is used to normalize the data such that the resulting output vector is a probability distribution that sums up to 1.

Then we import pretrained model that was trained for sentiment analysis into object MODEL, model was trained on twitter comments that were labeled (supervised learning) 

Test Roberta with example text from data set
Encoded text is transformed into binary for the algorithm to comprehend the data 
Run model on encoded object that created a tensor 
Transform tensor into numpy object for local storage purposes
Apply normalization function softmax 
Scores dictionary created for neg/neu/pos
Run on entire dataset (create a function + for loop)
Combine result  dictionaries  of both models for comparison models 

Compare Scores between models
Create pairplot matrix to compare neg/neu/pos of VADER and ROBERTA

Review Examples
Look for anomalies where the compound score is high (positive text detected but score is low, which means contradiction) 
Return the biggest contradiction example from dataset
FInd reasons for disparity 
Reason was nuanced sentenced (started off positive, but ended negatively) 

The Transformers Pipeline
Easiest 
Simplified 

Results 
Comparing models
According Fig 4 - Model comparison using pairplots:
VADER model less confident
ROBERTA model further to the right 
ROBERTA model has a greater degree os separation between neg/neu/pos scores 

Looking for anomalies 
VADER mis categories the most due to ‘bag of words approach’
Both models struggle with jokes, tongue in cheek, sarcasm and nuanced sentences 

. Conclusion 
Use transformer, due to smaller dataset, for sentiment analysis. In general, if you have a large and complex dataset, RoBERTa may be more accurate due to its advanced architecture and improved training process. However, if you have a smaller dataset or simpler task, the Transformer pipeline may be more suitable and easier to use. It's recommended to experiment with both models and choose the one that performs best on your specific task and dataset.

