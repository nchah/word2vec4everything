# word2vec4everything

Processing some interesting text documents through the word2vec machine learning model.

## word2vec

`word2vec`, as explained in [Wikipedia](https://en.wikipedia.org/wiki/Word2vec), refers to a number of machine learning models that accept a body of text and outputs a vector space of word embeddings.
The resulting word vectors can be visualized in such a way that words with similiar semantic meanings and contexts are clustered together.

As word2vec is an unsupervised machine learning technique, the input text source does not require any labels.


## Dependencies

This project implements word2vec using Google's TensorFlow library.

- Python 2.x
- TensorFlow

The TensorFlow scripts in the `python` directory are modifications to the starter code provided in the TensorFlow tutorials.
Modifications include:

- Python PEP8 styling changes
- General refactoring 
- Further code to adjust the visualization step


## Gallery

**All 7 Harry Potter books**
- Data: ~10 MB - A plaintext file of all 7 Harry Potter books. Found with the help of some Google-fu.
- Comment: word2vec clusters the 4 houses of Hogwarts (Gryffindor, Hufflepuff, Ravenclaw, and Slytherin) together.
![](images/tsne-hp-names-200k-steps-1500-plot-v2-houses-cluster.png)







