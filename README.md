# Luchan - A Framework for Analyzing Communication according to Niklas Luhmann on 4chan Using Graph Measures and Topic Modelling

This is the repo where I present the code I use during my masters thesis. The goal of the thesis is to research if there are communication structures that are reproducing themselves within similar topics. The theoretical background for this research is given by the assumptions of Niklas Luhmann on communication within social systems.

It contains:
- the scraper I used to collect the data from the 4chan API
- the extractor for mining the relevant data from the text, like the id's used to create the thread networks, the quotes etc.
- the graph_pipeline for extracting the network features
- the nlpipe for creating the topic model (NlPipe_sk.py contains the old implementation in sklearn. I switched to gensim due to their implementation of the coherence score for testing the quality of the lda model.)
- not implemented yet: code for subgroup discovery
- maybe also the data I collected

At the beginning of the thesis I only had access to a limited data set (~160k threads from 4chan.org/b/). Due to that, I opted for topic modelling and common graph measures. Specially for the latter, this also enables a better interpretability, compared to embedding methods. During my exploration I also collected about 1m-1.5m additional threads from several boards. Also another dataset containing ~3.3m threads from 4chan.org/pol/ got released. With this amount of data, I can also consider graph and document embeddings, if this doesn't go beyond the scope of a masters thesis.




