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

#### Removing label redundancy:
- As noted in our midterm report, we had a lot of redundant labels that had a lot of overlap and confused the models, dropping their accuracy. So after inspecting the confusion matrix we found labels that were similar and had a lot of overlap and combined them as one distinct label. Performing this provdied a blanket accuracy increase of ~10% for all our models. We investigate the confusion matrices before and after for each model in the model metrics section to get a better understanding. We reduced unique labels from 23 down to 13 unique labels.


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

#### Model 5- Unsupervised Learning: Latent Dirichlet Analysis (gensim model's LdaModel)

Why LDA?
- Generative soft clustering approach especially relevant for text analysis.
- Documents are represented as a mixture of topics where a topic is a bunch of words. Topics are considered as latent variables as the topics are inferred from observed document-word frequencies. Hence, LDA looks at a document to determine a set of topics that are likely to have generated that collection of words.
- Output of LDA could be used in implementing an automated chatbot for customer care purposes where form the uesr's prompt it infers the topic (financial service in our case) and provides targeted help for the financial service.

## Results and Discussion

Since the entire dataset is too large, we have limited our models to process and train on a shorter dataset of 10000 records of each category. 

### Exploratory Data Analysis (EDA)

#### Distribution of Labels in the dataset before combining:
<img width="900" alt="Distribution - Labels" src="https://github.com/v-divyansh1/ProjectProposal/assets/157415627/2b9c336e-8261-4954-a920-65ad81d2176d">

#### Distribution of Labels in the dataset after combining:
![label_distribution_after_combining](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/367bfc4e-67dc-448f-a8ea-896dd1b45f93)


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

Before combining labels:
- Accuracy Score: 63.4%
- F1 score: 62.6%
- Precision: 63.0%
- Recall: 63.4%

After combining labels:
- Accuracy: 74.8%
- F1 Score: 74.3%
- Precision: 74.2%
- Recall: 74.8%

Confusion Matrix before combining labels:
![SVM - confusion matrix](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/5cdf3789-57d1-4440-9d9b-2bfb0d1af4a3)

Confusion Matrix after combining labels:
![new_RF_conf](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/3fcfe07f-7629-4c68-810f-c0cc4597110c)

We can see a drastic reduction in misclassification in before and after. Now our model is able to classify more accurately.

#### Model 2: Random Forest

Before combining labels:
- Accuracy Score: 60.8%
- F1 score: 59.1%
- Precision: 60.9%
- Recall: 60.8%

After combining labels:
- Accuracy: 70.8%
- F1 Score: 68.8%
- Precision: 69.3%
- Recall: 70.8%

Confusion Matrix before combining labels:
![RF - confusion matrix](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/dd779b21-71fc-454a-b71b-1756b570f6c1)

Confusion Matrix after combining labels:
![new_RF_conf](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/29855ab5-40eb-492a-9f71-4e7f4ea20888)

We can see a drastic reduction in misclassification in before and after. Now our model is able to classify more accurately.

#### Model 3: XGBoost Classifier

Before combining labels:
- Accuracy Score: 62.0%
- F1 score: 61.7%
- Precision: 62%
- Recall: 62.3%

After combining labels:
- Accuracy Score: 74.3%
- F1 score: 73.6%
- Precision: 73.5%
- Recall: 74.3%

Confusion Matrix before combining labels:
![Confusion_Matrix_XGB](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/034b900d-2d2e-4a40-a6c5-effd79b3cd79)

Confusion Matrix after combining labels:
![xgboost_after_combining_confusion_matrix](https://github.com/ahv15/ProjectProposal/assets/157415627/b0beec6f-ffd1-44b8-8f22-c7a64e92b81c)


We can see a drastic reduction in misclassification in before and after. Now our model is able to classify more accurately.

#### Model 4: KMeans

PCA was performed on top 2 components for visualisation purposes. We chose 5 clusters for now.

![KMEANS_MIDTERM](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/ce3a302f-a161-4025-a897-02f04ac1cea1)

#### Model 5: LDA

Top words from each cluster can represent a topic (financial service for our use case).
![LDA_CLusters1](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/5337d84c-3947-499e-9c12-e638c3e52697)
![LDA_Clusters2](https://github.com/v-divyansh1/ProjectProposal/assets/157415627/2ef249be-881b-4734-a9a8-d16007dd5961)

### Model 
- SVMs work best for high-dimensional feature spaces, making them suitable for NLP tasks where feature dimensions are large. However, they struggle with large datasets due to scalability issues, and they require careful parameter tuning, which can be challenging due to the huge size of data. In our project as well, we can see that SVM shows the highest accuracy score. 
-	RF's ensemble approach can capture complex relationships in text data where feature interactions are important. However, they may lack interpretability compared to SVMs, and training time can be significant for large datasets. 
-	XGBoost's high performance and built-in regularization techniques make it a good choice, especially when feature importance and generalization are crucial. That is why, it shows a high accuracy score. However just like SVM, it takes a long time to train and requires careful hyperparameter tuning.
   
### Future Work

- Hyperparameter tuning of current models to get better accuracy metrics. Even with current smaller sized dataset tuning takes infeasibly long and would require more advanced resources which can be picked up in the future.
- Investigation with unsupervised models if we are able to uncover behavioral patterns based on words within cluster used for each financial product. Could help in redesign and design of old and new products.

## References

1. T. Mikolov, “Efficient estimation of word representations in vector space,” arXiv.org, Jan. 16, 2013. https://arxiv.org/abs/1301.3781
   
2. J. Polpinij and B. Luaphol, “Comparing of multi-class text classification methods for automatic ratings of consumer reviews,” in Lecture Notes in Computer Science, 2021, pp. 164–175. doi: 10.1007/978-3-030-80253-0_15

3. M. Goudjil, M. Koudil, M. Bedda, and N. Ghoggali, “A novel active learning method using SVM for text classification,” International Journal of Automation and Computing, vol. 15, no. 3, pp. 290–298, Jul. 2016, doi: 10.1007/s11633-015-0912-z

4. I.-C. Chang, T. Yu, Y. Chang, and T. Yu, “Applying text mining, clustering analysis, and latent dirichlet allocation techniques for topic classification of environmental education journals,” Sustainability, vol. 13, no. 19, p. 10856, Sep. 2021, doi: 10.3390/su131910856

 
## Contribution Table

| Name             | Final Report Contributions |
|---------         |------------------------|
| Divyansh Verma   | Final Report creation |
| Rajani Goudar    | Reducing label redundancy and All models comparison |
| Harshit Alluri   | LDA unsupervised model Selection, Design and Implementation |
| Jyothsna Karanam | Video creation and recording  |
| Nikita Agrawal   | Reducing label redundancy and All models comparison |


## Gantt Chart
![Gantt_Chart_final](https://github.com/ahv15/ProjectProposal/assets/157415627/cd183090-5827-45d2-9ae5-333ad475c824)




