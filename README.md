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
- Tokenization (NLTK's word_tokenize): For efficient segmentation of complaint text into tokens.
- Stemming/Lemmatization (NLTK's PorterStemmer/WordNetLemmatizer): To normalize words and reduce dimensionality.
- Part-of-speech tagging (NLTK's pos_tag function): To identify the grammatical components of each token.
- Stopword and Duplicate Word Removal: They will be removed using NLTK's built-in stopwords list and Python's set operations.

### Machine Learning Models
These supervised models are well-suited for text classification tasks, offering high accuracy and robustness:
- Support Vector Machine (scikit-learn's SVC)
- XGBoost (xgboost library)

Additionally, the following unsupervised methods are employed to identify latent topics and group similar complaints without the need for labeled data:
- Latent Dirichlet Allocation (scikit-learn's LDA)
- Clustering (scikit-learn's KMeans) 

## Results and Discussion
- To evaluate the effectiveness of our proposed methodology, we will utilize quantitative metrics such as F1 score, accuracy, precision and recall.
- Our project goals include achieving high accuracy in classifying complaints into their respective categories and ensuring balanced precision and recall across all labels.
- Through rigorous evaluation of our methodology, we aim to develop a robust/generalizable solution for automated complaint tagging in the domain of customer service to enhance efficiency and improve customer satisfaction.


## References

1. T. Mikolov, “Efficient estimation of word representations in vector space,” arXiv.org, Jan. 16, 2013. https://arxiv.org/abs/1301.3781
   
2. J. Polpinij and B. Luaphol, “Comparing of multi-class text classification methods for automatic ratings of consumer reviews,” in Lecture Notes in Computer Science, 2021, pp. 164–175. doi: 10.1007/978-3-030-80253-0_15

3. M. Goudjil, M. Koudil, M. Bedda, and N. Ghoggali, “A novel active learning method using SVM for text classification,” International Journal of Automation and Computing, vol. 15, no. 3, pp. 290–298, Jul. 2016, doi: 10.1007/s11633-015-0912-z

4. I.-C. Chang, T. Yu, Y. Chang, and T. Yu, “Applying text mining, clustering analysis, and latent dirichlet allocation techniques for topic classification of environmental education journals,” Sustainability, vol. 13, no. 19, p. 10856, Sep. 2021, doi: 10.3390/su131910856

 
## Contribution Table

| Name    | Proposal Contributions |
|---------|------------------------|
| Divyansh Verma | Introduction and Problem Definition |
| Rajani Goudar | Video Recording |
| Harshit Alluri    | Github Page |
| Jyothsna Karanam    | Methods, Results and Discussion  |
| Nikita Agrawal    | Powerpoint Presentation |


## Gantt Chart
![GanttChart](https://github.com/ahv15/ProjectProposal/assets/52852877/3e26cf00-b6f4-4e1a-aa1c-830bc0cafb4c)


