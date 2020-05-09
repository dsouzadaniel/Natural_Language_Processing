# Hierarchical Attention Networks for Document Classification

This is a PyTorch implementation of the fantastic HAN Paper from CMU(( ***https://www.aclweb.org/anthology/N16-1174.pdf***)) 
complete with Yelp Dataloaders and Minibatching for Training

![HAN Architecture]("https://github.com/dsouzadaniel/Natural_Language_Processing/blob/master/Implementations/HAN/han_model_architecture.png")

The only modification is that I have swapped out GloVe for Elmo for improved performance

* Dataset *
I have uploaded a sample dataset from the Yelp 2013 Reviews Dataset.
Download the complete dataset at [Yelp Reveiws Dataset](https://www.yelp.com/dataset/download)


I have also implemented a Streamlit App to interact with the model! ^_^

![Streamlit App]("https://github.com/dsouzadaniel/Natural_Language_Processing/blob/master/Implementations/HAN/han_model_app.png")


* To install requirements

`pip install -r requirements.txt`

* To train a new model

` python train.py`

* To run the Demo Streamlit App
 
` streamlit run app.py`
