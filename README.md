## Abstractive Text Summarization using Sequence-to-sequence RNNs and Beyond
Encoder-decoder model with attention mechanism
### Word embedding

We used the [Glove](https://nlp.stanford.edu/projects/glove/) pre-trained vectors to initialize the word embeddings.

### Encoder
Bidirectional GRU-RNN.
### Decoder
Unidirectional GRU-RNN, with beamsearch.

### Attention Mechanism
We Used BahdanauAttention.
### Data 
Data included in our github is a reduced dataset extracted from the dataset available at [harvardnlp/sent-summary](https://github.com/harvardnlp/sent-summary).

### Evaluation
We used the ROUGE metric, from the package [py-rouge](https://pypi.org/project/py-rouge/).

### Requirements
- Python 3
- Tensorflow version 1.x
- pip install -r requirements.txt

if google colab every dependency is installed in the notebook.
### Usage

To use our implementation you simply go to the notebook [text_summarization_feats.ipynb](https://github.com/devhemza/deeplearningproject/blob/main/text_summarization_feats.ipynb). And run the cells.

### References:

1.  D. Bahdanau, K. Cho, Y. Bengio, Neural machinetranslation by jointly learning to align and trans-late, arXiv preprint arXiv:1409.0473.

2. C.-Y. Lin, ROUGE: A package for automatic eval-uation  of  summaries,   in:    Text  SummarizationBranches Out, Association for Computational Lin-guistics, Barcelona, Spain, 2004, pp. 74–81.URLhttps://www.aclweb.org/anthology/W04-1013.

3. R.   Nallapati,   B.   Zhou,   C.   Gulcehre,   B.   Xi-ang,  et  al.,  Abstractive  text  summarization  us-ing sequence-to-sequence rnns and beyond, arXivpreprint arXiv:1602.06023
