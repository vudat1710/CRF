# AI Assignment - Named Entity Recognition using Conditional Random Fields
Project attemps to train a CRF model for Named Entity Recognition (NER) problem using data from VLSP 2016 and crfsuite library.

# Problem Description

In general, NER is one of the most fundamental problems in Natural Language Processing (NLP), especially for the purpose of extracting invaluable information from text in order to identify and classify named entities. These entities could be some predefined categories such as Person, Organization, Location, etc. These specific terms could possibly be defined differently depends on its surrounding words and the meaning of the whole sentences or paragraph. There are numerous researches conducted with years of experimenting, which results in several approaches to this problem. For instance, rule-based NER like NLTK's time expression tagger is one in many classical methods, or some machine learning approaches are also familliar for many researchers in this field. In order to take advantage of surrounding context and semantic labels of tokens in text input, a typical used method is conditional random field (CRF). It is a type of probabilistic undirected graphical model that can be used to model sequential data. CRF calculates the conditional probability of values on classified output nodes given values assigned to the classified input nodes. This project uses a powerful CRF library called CRFsuite (http://www.chokkan.org/software/crfsuite/manual.html) with training and test data from VLSP 2016 to have deeper insights into NER problem.

# System requirements
Python 2+ or 3+ with sklearn_crfsuite, sklearn, eli5 and pickle packages installed

# Compilation Steps
1. Modeling:

- cd to the CRF directory.
- In this directory, open terminal and run the script `python CRF.py` or `python3 CRF.py` depends on the version of python you are using.
- After this step we can have a model called "finalized_model.pkl" and "weight.html" file (this html file show the transition and state matrices learned from training set)

2. Get output result:

- Compile and run the code in LoadModel.py (similar to the previous step). The output file is "output.txt"
- In this directory, we have the script provided by my supervisor 'conlleval.pl' (written by Thai Hoang Pham - https://github.com/pth1993/NNVLP/blob/master/conlleval.pl). Run the script by typing `./conlleval.pl -l -d '\t' < output.txt` to get token-based result. Run the script by typing `./conlleval.pl -l -r -d '\t' < output.txt` to get chunk-based result. (on windows, install perl then run the script `perl conlleval.pl -l -d "\t" < output.txt` and `perl conlleval.pl -l -r -d "\t" < output.txt` on cmd to get corresponding results).

3. Demo

- Compile and run the code in demo.py
- In terminal, enter any Vietnamese text
- Press enter, the result will appear on screen
  This is a very simple demo and still cannot solve the problem of end punctuation syllables.
