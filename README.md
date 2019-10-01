# The Politicker

What are your political representatives saying?

## What is The Politicker?

Almost daily, your Members of Parliament make decisions on your behalf.


## How do you use The Politicker?

Enter the name of a Canadian Member of Parliament (MP), followed by a topic that you would like to learn more about.


Soon afterwards, you will see a summary of all that he or she said on that topic during parliamentary debates.


Underneath the summary, you can explore all of the speeches that were considered in creating the summary.


## How does The Politicker work?

The Politicker makes use of Hansard transcripts of Canadian federal parliamentary debates. These transcripts are publicly available through the House of Commons [website](https://www.ourcommons.ca/en).

There are two input fields on the home page of The Politicker. In the first field, the user enters the name of a Member of Parliament (MP), with autocomplete suggestions available if needed. The page then updates with an image of the selected MP.

In the second field, the user enters a topic of interest, where autocomplete suggestions are limited to those discussed by the MP in the first input field. Based on the entered topic, the server concatenates all speeches by the selected MP, in reverse chronological order. It then tokenizes the speeches into sentences, and passes the tokens to an extractive summarization algorithm. Note that an extractive summary consists of sentences that appear in the original speeches. This is in contrast to an abstractive summary, which consists partly (or entirely) of newly generated sentences. The extractive approach seemed less likely to 'put words in someone's mouth'.

The Politicker's extractive algorithm performs four steps: 1) Encoding the tokenized sentences using a language representation model called Bidirectional Encoder Representations from Transformers, or BERT (see Appendix below for more information), 2) applying K-Means clustering to the encoded tokens, 3) selecting the sentences closest to the centroid of each cluster and, 4) concatenating them to form a summary. The server then updates the HTML (with the help of Javascript functions) to include the summary, alongside all of the speeches that were provided to the summarization algorithm (in reverse chronological order).


## Appendix

### BERT
The original BERT [publication](https://arxiv.org/abs/1810.04805).
