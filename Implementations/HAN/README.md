# Hierarchical Attention Networks for Document Classification

This is a PyTorch implementation of the fantastic HAN Paper from CMU(***https://www.aclweb.org/anthology/N16-1174.pdf***) 
complete with Yelp Dataloaders and Minibatching for Training

![HAN Architecture]("https://i.ibb.co/ygb4h6q/han-model-architecture.png")

The only modification is that I have swapped out GloVe for Elmo for improved performance

* Dataset *
I have uploaded a sample dataset from the Yelp 2013 Reviews Dataset.
Download the complete dataset at [Yelp Reviews Dataset](https://www.yelp.com/dataset/download)


I have also implemented a Streamlit App to interact with the model! ^_^

![Streamlit App]("https://i.ibb.co/z4V303H/han-model-app.png")


* To install requirements

`pip install -r requirements.txt`

* To train a new model

` python train.py`

* To run the Demo Streamlit App
 
` streamlit run app.py`
