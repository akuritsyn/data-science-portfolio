# Data Science Portfolio


## **Traditional Machine Learning**

### 1. [Kaggle: Customer Churn Prediction](https://github.com/akuritsyn/Machine_Learning_Projects/tree/master/Customer_Churn_Prediction) - top 5%
The goal of the Customer Churn Prediction project is to find clients of a telecom company who are likely to churn. Banks, telephone service companies, Internet service providers, pay TV companies, insurance firms, alarm monitoring services and others often use customer churn analysis and customer churn rates as one of their key business metrics, because the cost of retaining an existing customer is far less than acquiring a new one. Companies from these sectors often have customer service branches, which job is to win back defecting clients, because long-term clients can bring much more profit to a company than newly recruited ones.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In this project, we make a prediction that a client will stop using a service during a pre-defined time period by solving a binary classification problem. We'll be using data from the 2009 competition described [here](http://www.kdd.org/kdd-cup/view/kdd-cup-2009/Intro). This is a real data set collected by the French telecom company Orange, where all the personal information about users has been deleted so that individual users can not be identified based on this data. The data set consists of 50,000 objects and comprises 190 numerical and 40 categorical variables. We'll use 40,000 objects as a training data set and remaining 10,000 as a validation data set. The data has significant imbalace, where less that 8% of users belong to the churn class.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; In practice, using these predictions one can determine clients who are likely to churn and take measures to keep them as well as to idenify and fix any existing problems. In particular, model developed here can be used as a baseline for defining priorities of the customer retention campaign in the framework of a chosen economic model.\
**Keywords:** Classification, feature engineering, one-hot encoding, scikit-learn, pandas, XGBoost, ROC AUC
</br>
</br>
### 2. [Kaggle: Identification of internet users based on their internet behavior](https://github.com/akuritsyn/Machine_Learning_Projects/tree/master/User_Identification) - top 9%
In this project we are solving a problem of user identification based on their behavior on the Internet. This is a complex and interesting problem at a junction of data science and behavioral phsycology. For example, many email providers are trying to identify email hackers based on their behavior. Obviously, a hacker will behave differently than the email owner: he may not be deleting emails after reading them as the email owner used to do, he will be marking messages differently, etc. If this is a case, such an intruder can be identified and expelled from the email box by offering the true owner to enter his email acccount by using a code sent via SMS. In data science such problems are called "Traversal Pattern Mining" or "Sequential Pattern Mining". Such problems are solved, for example, by a Google analytics team.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Here, we will attempt to identify a person based on data about websites sequentially visited by him/her. The idea is that users move differently between websites and this can help to identify them.\
**Keywords:** Classification, feature engineering, one-hot encoding, scikit-learn, pandas, XGBoost, Vowpal Wabbit, tf-idf
</br>
</br>
### 3. [Kaggle: House Prices](https://github.com/akuritsyn/kaggle-house-prices)
Built a a model to predict prices of the residential homes in Ames, Iowa. Employed a tabular model from the factai v1 library running on top of PyTorch to perform data preprocessing, feature generation and a two-layer fully connected neural network to generate predictions in a semi-automated way.\
**Keywords:** Regression, EDA, feature engineering, time features, categorical features, AutoML, fastai, pandas, neural network
</br>
</br>
### 4. [Annual Income Prediction](https://github.com/akuritsyn/Machine_Learning_Projects/tree/master/Income_Prediction)
The goal of this Kaggle competition organized by Jigsaw and Google is to build a model that recognizes toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion.\
**Keywords:** Classification, EDA, feature engineering, kNN, scikit-learn, pandas, fastai
</br>
</br>
### 5. [Kaggle: ASHRAE - Great Energy Predictor III](https://www.kaggle.com/c/ashrae-energy-prediction) - top 19% (679/3614)
Developed models to predict metered building energy usage in the following areas: chilled water, electric, hot water, and steam meters.\
**Keywords:** EDA, feature engineering, scikit-learn, pandas, XGBoost, light GBM
</br>
</br>
### 6. [Coursera: Forecast mean wages in Russia](https://github.com/akuritsyn/Machine_Learning_Specialization/blob/master/5%20-%20Applied%20Problems%20in%20Data%20Science/PA/5PA_1_Wage_Forecast.ipynb)
Developed a model using SARIMA's time series approach to forecast mean wages in Russia.\
**Keywords:** Time series, EDA, feature engineering, SARIMA, pandas
</br>
</br>
</br>

## **Natural Language Processing**

### 1. [Kaggle: Jigsaw Unintended Bias in Toxicity Classification](https://github.com/akuritsyn/kaggle-jigsaw) - top 1% (17/3165)
The goal of this Kaggle competition organized by Jigsaw and Google is to build a model that recognizes toxicity in online conversations, where toxicity is defined as anything rude, disrespectful or otherwise likely to make someone leave a discussion.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The first developed toxicity models incorrectly learned to associate the names of frequently attacked identities with toxicity. Models predicted a high likelihood of toxicity for comments containing those identities (e.g. "gay"), even when those comments were not actually toxic (such as "I am a gay woman"). This happens because training data was pulled from available sources where unfortunately, certain identities are overwhelmingly referred to in offensive ways. Training a model from data with these imbalances risks simply mirroring those biases back to users.\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; We were challenged to build a model that recognizes toxicity and minimizes this type of unintended bias with respect to mentions of identities using a dataset labeled for identity mentions and optimizing a metric designed to measure unintended bias.\
**Keywords:** Classification, NLP, BERT, RNN, LSTM, embeddings, model ensembling, PyTorch, Nvidia GPU
</br>
</br>
### 2. [Coursera: Stackoverflow Assistant NLP Chatbot Accessible via Telegram](https://github.com/akuritsyn/Machine_Learning_Projects/tree/master/StackOverflowAssistantBot)
Built a chatbot running on AWS and accessible via Telegram messenger. The chatbot is able to (1) answer programming questions using Stackoverflow dataset and (2) chit-chat and simulate dialogue on all non-programming related questions.\
**Keywords:** Classification, NLP, bag-of-words, tf-idf, embeddings, cosine similarity, scikit-learn, AWS EC2, Telegram, Docker. 
</br>
</br>
### 3. [Udacity: Movie Sentiment Analysis Web App](https://github.com/akuritsyn/udacity-ml-nanodegree/tree/master/sentiment-analysis-model)
Built a web app hosted on AWS to determine movie review sentiment. Converted text reviews into numerical representation and built a model is using LSTM classifier in PyTorch.\
**Keywords:** Classification, NLP, RNN, LSTM, scikit-learn, PyTorch, AWS Sagemaker, AWS Lambda, Amazon S3, Amazon API Gateway
</br>
</br>
### 4. [Udacity: Plagiarism Detector](https://github.com/akuritsyn/udacity-ml-nanodegree/tree/master/plagiarism-detection)
Built a plagiarism detector that examines a text file and performs binary classification, labeling that file as either plagiarized or not, depending on how similar the text file is when compared to a provided source text. Extracted similarity features from texts based on common n-gram counts (containment) and longest common subsequence (LCS) and employed a random forest classifier trained on AWS Sagemaker.\
**Keywords:** Classification, NLP, n-grams, Random Forest,  AWS Sagemaker, Amazon S3
</br>
</br>
### 5. [Coursera: Movie Sentiment Analysis Web App #2](https://github.com/akuritsyn/udacity-ml-nanodegree/tree/master/sentiment-analysis-model)
Built a web app hosted on AWS to determine movie review sentiment. A pipeline consisting of a CountVectorizer, tf-idf and LinearSVC is employed to build a classifier, then a standard ntlk data set is used for classifier training, and finally a web demo is created using flask.\
**Keywords:** Classification, NLP, scikit-learn pipeline, nltk, tf-idf, Flask, AWS EC2
</br>
</br>
</br>

## **Computer Vision**

### 1. [Kaggle: Recognizing Faces in the Wild](https://github.com/akuritsyn/kaggle-recognizing-faces) - top 1% (4/458)
The goal of this competetion is to determine if two people are blood-related based solely on images of their faces. Built a classifier using a VGGFace siamese CNN model with a modified architecture.\
**Keywords:** Image classification, Siamese CNN,VGG-Face, ResNet50, Keras, Nvidia GPU
</br>
</br>
### 2. [Kaggle: SIIM-ACR Pneumothorax Segmentation](https://github.com/akuritsyn/kaggle-recognizing-faces) - top 4% (50/1475)
Developed a model to classify (and if present, segment) pneumothorax (collapsed lungs) from a set of chest radiographic images. The final model is based on an ensemble of Unet models with pretrained (on ImageNet) EfficientNetB4 and ResNet34 encoders in Keras and PyTorch averaged over folds.\
**Keywords:** Image segmentation, CNN, Unet, augmentations, EfficientNetB4, ResNet34, ensembling, Keras, PyTorch, Nvidia GPU
</br>
</br>
### 3. [Kaggle: RSNA Intracranial Hemorrhage Detection](https://github.com/akuritsyn/kaggle-rsna-intracranial-hemorrhage) - top 5% (56/1345)
Built a model based on ResNext50_32x4d pretrained on ImageNet to detect acute intracranial hemorrhage and its subtypes.\
**Keywords:** Image classification, CNN, augmentations, ResNext50_32x4d, ensembling, PyTorch, Nvidia GPU
</br>
</br>
### 4. [Kaggle: Severstal: Steel Defect Detection](https://www.kaggle.com/c/severstal-steel-defect-detection) - top 8% (181/2431)
Built a model to classify surface defects in steel.\
**Keywords:** Image classification, CNN, augmentations, ensembling, PyTorch, Nvidia GPU
</br>
</br>
### 5. [Kaggle: TGS Salt Identification](https://github.com/akuritsyn/TGS_Salt_Identification_Challenge) - top 12% (368/3229)
Developed a segmentation model to identify salt deposits beneath the Earth's surface based on seismic images  using Unet CNN.\
**Keywords:** Image segmentation, CNN, Unet, augmentations, ResNet34, ensembling, Keras, Nvidia GPU




