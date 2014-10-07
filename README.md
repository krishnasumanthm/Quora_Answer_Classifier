Quora_Answer_Classifier
=======================

Quora uses a combination of machine learning (ML) algorithms and moderation to ensure high-quality content on the site. High answer quality has helped Quora distinguish itself from other Q&A sites on the web. The task here is to devise a classifier that is able to tell good answers from bad answers, as well as humans can.  A good answer is denoted by a +1 in our system, and a bad answer is denoted by a -1.  

In this work I implemented several models (Naive Bayes, Logistic Regression, Random Forests, AdaBoost, SVM, LDA, QDA, Decision Trees) with different feature selection methods such as feature scaling, Lasso feature selection and Linear feature selection and compared there accuracy.

More information about the Quora Answer Classifier challenge can be obtained [here](https://www.quora.com/challenges#answer_classifier).

Simulation
=======================
The execute.py is the entry point of the simulation. The data sets are present in data folder. The extract.py extracts the features and labels for the train and test data from the input and output files. to run the simulation use command :-
```python
 python execute.py
```
Note that the simulation requires the libraries scikit-learn and numpy.
