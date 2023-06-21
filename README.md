# WeiboGDPMapping
This repository contains the code to reproduce the paper [__Predicting Economic Development Using Geolocated Wikipedia Articles__](https://dl.acm.org/citation.cfm?doid=3292500.3330784).

# Preparation

## Weibo dataset
We use [web crawler](https://github.com/dataabc/weiboSpider) to collect articles published on [Weibo](https://weibo.com/) by users from different provinces in China.  

All the articles collected is in `./data/weibo`. The first-level directory name is the province where the publisher is located, the second-level directory name is the publisher's name, and the file name is the publisher's Weibo ID.

## Night light dataset
We collected a night light image dataset from [WorldView](https://worldview.earthdata.nasa.gov/). Specifically, we collected night light images of different sizes for each city on different dates with the capital cities as the center. Due to other factors such as weather and the significant differences in latitude and longitude among China's provinces, the night light images collected from different provinces at the same time are affected differently. To reduce this impact, we collected night light images of different sizes and on different dates.
 
All the images collected is in `./data/nightlight_download`.

Code to collect is in `nightlight_download.py`

## Ground Truth data
Our ground truth data consists of per capita GDP, number of hospital beds per 10,000 people, and the Engel coefficient from [Chinese National Bureau of Statistics](https://data.stats.gov.cn/).

All the data collected is in `./data/groundtruth.csv`.

## Preprocess
We provide the code for extracting and processing data in `process.py` and `my_dataset.py`.

## Models
We provide the code for Nightlight only model and Multi-modal model in `models.py`.

# Training
Some superparameter can be changed in `config.py`

## Training Docvec on Weibo Articles
We use [gensim](https://radimrehurek.com/gensim/models/doc2vec.html) doc2vec packege for training the Doc2Vec model. To train the Doc2Vec model on geolocated articles, run:

```
python train_doc2vec.py
```

## Nightlight Only Model
To train the night light only model, run

```
python train_NLModel.py
```

## Multi-modal Model
To train the multi-modal model, run

```
python train_MultiModel.py
```
