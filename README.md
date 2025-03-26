# 🌐 Towards Website Domain Name Classification using Graph-Based Semi-Supervised Learning

This repository contains the official implementation of the experiments described in the paper:

**_Towards Website Domain Name Classification Using Graph-Based Semi-Supervised Learning_**

---

## 🧠 Project Description

The goal of this project is to classify website domain names using various machine learning and deep learning techniques, including:
- TF-IDF
- SVM with Word2Vec (fastText)
- LSTM
- Graph-based Semi-Supervised Learning
- NFA (Number of False Alarms)

The code performs full experiment automation including training, cross-validation, hyperparameter tuning, and final evaluation.

---

## 🗂️ Code Structure

### 📄 `main.py`
- Central script for running all experiments.
- Loads user activity dataset and domain vector file.
- Splits data into training and testing sets.
- Performs **10-fold cross-validation** for each method.
- Tunes parameters based on performance.
- Executes the best method on the test set.

---

### 📄 `TFIDF.py`
- Classifies domains using **TF-IDF** scoring.
- Evaluates the importance of each n-gram for category prediction.
- Calls `training_model.py` to build n-gram models.

---

### 📄 `NFA.py`
- Implements a **Number of False Alarms (NFA)**-based method for domain classification.
- Uses n-gram similarity scoring per category.
- Calls `training_model.py` to generate training models.

---

### 📄 `SVM.py`
- Uses **Word2Vec embeddings** (via [fastText](https://github.com/facebookresearch/fastText)) to represent domain names.
- Represents user sessions (domains visited within a 1-hour window) as input vectors.
- Applies **Support Vector Machines (SVM)** for classification.
- Requires precomputed domain vectors (`vectordomain.txt`).

---

### 📄 `Semi_Supervised_Graph.py`
- Builds graphs from domain names, sessions, or both using the `Create_Graph` function.
- Applies **graph-based semi-supervised learning**.
- Calls `semi_supervised.py`, which implements **LGC (Learning with Local and Global Consistency)** for classification.

---

### 📄 `LSTM.py`
- Runs an **LSTM model** from [Keras](https://www.tensorflow.org/guide/keras/rnn).
- Treats each domain name as a sequence of characters and uses character-level features for classification.

---
## 📜 Citation & Reference

If you find this work helpful, please consider citing our paper:

**Towards Website Domain Name Classification Using Graph-Based Semi-Supervised Learning**  
*Computer Networks*, **Volume 188**, 7 April 2021, Article 107865  
[🔗 Read on ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1389128621000384)

### BibTeX

```bibtex
@article{faroughi2021graph,
  title     = {Towards Website Domain Name Classification Using Graph-Based Semi-Supervised Learning},
  author    = {Azadeh Faroughi, Andrea Morichetta, Luca Vassio, Flavio Figueiredo, Marco Mellia and Reza Javidan},
  journal   = {Computer Networks},
  volume    = {188},
  pages     = {107865},
  year      = {2021},
  issn      = {1389-1286},
  doi       = {10.1016/j.comnet.2021.107865},
  url       = {https://www.sciencedirect.com/science/article/abs/pii/S1389128621000384}
}
```
---

## 🛠️ Requirements

Tested with **Python 3.7**

Required libraries:

- `scikit-learn`  
- `scipy`  
- `numpy`  
- `nltk`  
- `keras`  
- `torch`  

You can install them using:

```bash
pip install -r requirements.txt
```
