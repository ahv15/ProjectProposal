# Customer Complaint Classification

## Introduction
NLP involves the development of algorithms and models to enable computers to understand, interpret, and generate human language. It encompasses a wide range of tasks like text classification, named entity recognition, sentiment analysis etc. NLP techniques are applied for virtual assistants, information retrieval etc. We will use NLP to perform multiclass text classification to tag complaints to product/service.
### Literature Review
- [[1](#references)] This paper is the basis of Word2Vec text vectorization techniques that we will employ to perform word embedding. Word2Vec is a neural network framework that places similar words in a closer geometric space.
- [[2](#references)] Explores classification of consumer reviews using multiple supervised models like k-NN, MNB, RF, XGBoost and SVM. Similar to our topic and will provide guidance on our approach.
- [[3](#references)] Describes active learning approach for text categorization, aiming to reduce manual labeling effort by selecting informative samples through multi-class SVM classifiers, enhancing classification accuracy without compromising efficiency.
- [[4](#references)] To analyze environmental education research journals, Latent Dirichlet Allocation (LDA) is used to classify research topics. Results show that K-means and LDA methods largely align in identifying categories.

### [Dataset](https://catalog.data.gov/dataset/consumer-complaint-database)
Collection of complaints about financial products and services. Dataset has 18 columns and millions of records. There are 21 product and service labels. For our analysis we will be using a reduced dataset of around 200k records. We plan to use around 10k rows from each category to avoid class imbalance.

## Problem Definition
Customer issue resolution is one of the most important facets in running a smooth business operation and providing satisfactory services.  One of the biggest bottlenecks is identification and tagging of complaints to respective products/services. We can employ NLP techniques to parse complaint text and classify which product the complaint corresponds to, which automates the tagging process, thus reducing the need for human intervention and decreasing complaint resolution time.

## Methods

### Data Preprocessing Methods

#### Common preprocessing pipeline:
- Tokenization (NLTK's word_tokenize): For efficient segmentation of complaint text into tokens.
- Stopword and Duplicate Word Removal: They will be removed using NLTK's built-in stopwords list and Python's set operations.
- Lemmatization (NLTK's WordNetLemmatizer): To normalize words and reduce dimensionality.
- Part-of-speech tagging (NLTK's pos_tag function): To identify the grammatical components of each token.

#### Approach 1: Gensim Corpus Creation
- Creation of corpus dictionary using the preprocessed data from the previous step using corpora.dictionary() where word is mapped to integer ID's
- Use of corpus dictionary created to create bag-of-words representation in the form of (token_id, token_counts) representation where token_ids are queried from the dictionary.
- Used this approach to train our LDA __unsupervised__ model.

#### Approach 2: Creating word embeddings using Word2Vec and weighted average using TFIDF scores
- Creates a vector representation in a D dimensional space and similar context words are closer geometrically in the D dimensional hyperspace.
   - We used the pretrained "word2vec-google-news-300" model that converts a word to 300 dimensionality vectors.
- A weighted average based on TFIDF score of each word in an article (customer transcript for our use case) is used to represent the entire article as one 300 dimension vector.
- Performed Principal Component Analysis to reduce the articles to 2 dimensions only for visualization purposes. 
- Used this approach for our KMeans __unsupervised__ model. 

### Machine Learning Models Implemented

#### Model 1- Supervised Learning: Support Vector Machine (scikit-learn's SVM)

Why SVM?
- Popular supervised model with usually good performance on NLP tasks as seen in our literature analysis on the topic.
- Computationally efficient if using linear kernels and can be scaled to handle billions of documents with minimal processing delay.
- Performs well for text classification, where the input data is sparse and the feature space is typically high-dimensional.
- Less prone to overfitting compared to other classification algorithms.

#### Model 2- Supervised Learning: Random Forest (scikit-learn's KMeans)

Why Random Forest?
- Good for text classification, as it is capable of handling high-dimensional data efficiently and can effectively capture complex (even non-linear) relationships between features and labels.
- In text classification tasks, where the presence of irrelevant words or misspellings is common, Random Forest can effectively handle noise in the data without significantly impacting its performance.
- Reduces the risk of overfitting by averaging the predictions of multiple trees. By building multiple trees on random subsets of the data, Random Forest reduces the variance of the model and improves its generalization performance.
  
#### Model 3- Supervised Learning: XGBoost (scikit-learn's XGBClassifier)

Why XGBoost? 
- Excellently handles sparse data, common in text-based features, making it ideal for text classification.
- Utilizes advanced regularization (L1 and L2), gradient boosting, and tree pruning techniques to deliver highly accurate models, often outperforming other algorithms.
- Scales efficiently across multiple CPUs and GPUs for fast training times and supports various types of predictive modeling problems, including multi-class classification.

#### Model 4- Unsupervised Learning: KMeans Clustering (scikit-learn's KMeans)

Why KMeans?
- Simple unsupervised model that can be used as a baseline to compare the performance of more complicated unsupervised models like LDA.
- Low computation time.

## Results and Discussion

Since the entire dataset is too large, we have limited our models to process and train on a shorter dataset of 10000 records of each category. 

### Exploratory Data Analysis (EDA)

#### Distribution of Labels in the dataset:
<img width="900" alt="Distribution - Labels" src="https://github.com/v-divyansh1/ProjectProposal/assets/157415627/2b9c336e-8261-4954-a920-65ad81d2176d">

#### Distribution of length of customer complaint:
<img width="900" alt="Distribution - Length of Comment" src="https://github.com/v-divyansh1/ProjectProposal/assets/157415627/d2317b08-7e12-474a-8225-e9a60052e7e6">

This analysis could later on help us choose models for classification that might be limited by the length of text it can accept per comment. (Popular example: BERT, limited to 512 tokens)

#### Popular words before data preprocessing:
<img width="900" alt="Top 20 words pre - preprocessing" src="https://github.com/v-divyansh1/ProjectProposal/assets/157415627/39fc1463-a98a-4d8c-848a-b0a947e44bf0">

Clear evidence of data pollution, demonstrates the need for good data pre processing methods.

#### Wordcloud before data preprocessing:
<img width="900" alt="Word Cloud pre - preprocessing " src="https://github.com/v-divyansh1/ProjectProposal/assets/157415627/f4afd1b9-0023-4f6a-bb8e-4faad883d2ca">

#### Popular words after data preprocessing:
<img width="900" alt="Top 20 words after preprocessing" src="https://github.com/v-divyansh1/ProjectProposal/assets/157415627/9332a299-814f-4c1e-8413-73d12f6e704a">

Reduced data pollution and more accurate representation of popular words in the dataset.

#### Wordcloud after data preprocessing:
<img width="900" alt="Word Cloud after preprocessing " src="https://github.com/v-divyansh1/ProjectProposal/assets/157415627/aaf5708b-76c6-4ebf-9aaa-b515b900634f">

### Model Metrics

#### Model 1: SVM Classifier
- Accuracy Score: 63.4%
- F1 score: 62.6%
- Precision: 63.0%
- Recall: 63.4%

Confusion Matrix:
![SVM - confusion matrix](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/5cdf3789-57d1-4440-9d9b-2bfb0d1af4a3)

For a few labels that are similar ( eg label 1- credit card, label 2- credit card or prepaid card) the model is not able to differentiate them into the correct label. But as we can see, this is because the original labels have a lot of similarities and is not exclusive. This further supports our decision to merge similar labels into one category as outlined in the "Future Work" section.

#### Model 2: Random Forest
- Accuracy Score: 60.8%
- F1 score: 59.1%
- Precision: 60.9%
- Recall: 60.8%

Confusion Matrix:
![RF - confusion matrix](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/dd779b21-71fc-454a-b71b-1756b570f6c1)

#### Model 3: XGBoost Classifier
- Accuracy Score: 62.0%
- F1 score: 61.7%
- Precision: 62.04%
- Recall: 62.33%

Confusion Matrix:
![Confusion_Matrix_XGB](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/034b900d-2d2e-4a40-a6c5-effd79b3cd79)

Confusion matrix indicates a varied performance across different categories. For example, the model performs exceptionally well with 'Mortgage' and 'Student loan', where the classes have high precision and recall, suggesting the model can reliably identify and classify complaints in these categories. In contrast, categories like 'Debt or credit management' and 'Other financial service' have very low recall, indicating these are often incorrectly classified as other types, which could be due to insufficient representative data or similarities with other categories that confuse the model. The high precision in some categories but lower in others, combined with a general trend of moderate recall across most classes, suggests the model has learned to distinguish between the most prevalent classes effectively, but it struggles with less common or more nuanced categories. This behavior could be improved with more balanced training data, feature engineering to capture nuanced differences, or model tuning to better handle class imbalance.

#### Model 4: KMeans

PCA was performed on top 2 components for visualisation purposes. We chose 5 clusters for now.

![KMEANS_MIDTERM](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/ce3a302f-a161-4025-a897-02f04ac1cea1)

### Future Work

- Implementation of LDA unsupervised model and comparison of performance with KMeans.
- Hyperparameter tuning of current models to get better accuracy metrics.
- Accounting for skewed distribution of labels.
- We noticed that few labels in the original dataset are very similar. Could look into merging them into one common label in order to reduce total classes and see if it improves performance. Eg: "Credit card or prepaid card" and "Credit Card" are very similar categories and can be merged as one. "Payday loan, title loan, or personal loan" and "Payday loan, title loan, personal loan, or advance loan" are very similar too.

## References

1. T. Mikolov, “Efficient estimation of word representations in vector space,” arXiv.org, Jan. 16, 2013. https://arxiv.org/abs/1301.3781
   
2. J. Polpinij and B. Luaphol, “Comparing of multi-class text classification methods for automatic ratings of consumer reviews,” in Lecture Notes in Computer Science, 2021, pp. 164–175. doi: 10.1007/978-3-030-80253-0_15

3. M. Goudjil, M. Koudil, M. Bedda, and N. Ghoggali, “A novel active learning method using SVM for text classification,” International Journal of Automation and Computing, vol. 15, no. 3, pp. 290–298, Jul. 2016, doi: 10.1007/s11633-015-0912-z

4. I.-C. Chang, T. Yu, Y. Chang, and T. Yu, “Applying text mining, clustering analysis, and latent dirichlet allocation techniques for topic classification of environmental education journals,” Sustainability, vol. 13, no. 19, p. 10856, Sep. 2021, doi: 10.3390/su131910856

 
## Contribution Table

| Name             | Proposal Contributions |
|---------         |------------------------|
| Divyansh Verma   | EDA and Report |
| Rajani Goudar    | SVM Model |
| Harshit Alluri   | Data Preprocessing, Repo management |
| Jyothsna Karanam | XGBoost Model  |
| Nikita Agrawal   | Random Forest Model |


## Gantt Chart
![Gantt_Chart](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/8de233a2-85e2-4eab-afcd-0e94476ef2d8)


