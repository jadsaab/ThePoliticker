import pandas as pd
from gensim.summarization.summarizer import summarize
import matplotlib.pyplot as plt
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.cluster import KMeans
from nltk import tokenize
from lxml import html
import requests
import re

num_sentences = 4

# Load embedder
embedder = SentenceTransformer('bert-base-nli-mean-tokens')

def load_dataframe(path):
    """
    Function that loads the pandas dataframe containing the cleaned Hansard
    parliamentary debates
    """

    df = pd.read_pickle(path)
    return df


def get_img_urls(root_url):
    """
    Function that gets URLs to images of each MP on ourcommons.ca
    Typical root_url:
    "https://www.ourcommons.ca/Parliamentarians/en/members?page="
    """

    # Images of MPs are distributed across several pages
    # Find how many pages there are
    page = requests.get(root_url + '1')
    tree = html.fromstring(page.content)
    paging_btn_grp = tree.xpath('//nav[@class="paging btn-group"]/div/ul/li/a/@href')
    num_of_pages = int(re.findall('=\d*', paging_btn_grp[-1])[0][1:])

    # Loop through pages and store URLs to images
    pic_urls_across_pages = {}
    is_successful = True
    for i in range(num_of_pages):

        # Parse this page
        url = root_url + str(i+1)
        page = requests.get(url)
        tree = html.fromstring(page.content)

        # Store names and URLs from MP profiles on this page
        pic_urls = tree.xpath('//img[@class="picture"]//@src')
        first_names = tree.xpath('//span[@class="first-name"]/text()')
        last_names = tree.xpath('//span[@class="last-name"]/text()')

        if len(pic_urls) == len(first_names) and len(pic_urls) == len(last_names):
            # Update dictionary with names and URLs on this page
            full_names = [(first_names[i] + ' ' + last_names[i]) for i in range(len(pic_urls))]
            pic_urls_across_pages.update({full_names[i]: 'https:' + pic_urls[i] for i in range(len(pic_urls))})
        else:
            print('Mismatch of information when parsing MP profiles (%d pictures, %d first names, %d last names), aborting.\n' %
                  (len(pic_urls), len(first_names), len(last_names)))
            is_successful = False
            break

    if is_successful == False:
        pic_urls_across_pages['Placeholder'] = '../static/img/placeholder.png'

    return pic_urls_across_pages


def get_possible_speakers(df):
    """
    Function that returns a list of all speakers in dataframe
    """

    speakers = df['speakername']
    possible_speakers=[]
    for index, value in speakers.items():
        possible_speakers.append(value)

    return list(set(possible_speakers))


def get_speaker_topics(df, speakername):
    """
    Function that returns a list of all subtopics discussed by a
    user-specified speaker
    """

    df['subtopic'].fillna(' ', inplace=True)
    speaker_topics = df['subtopic'][df['speakername'] == speakername]
    possible_topics=[]
    for index, value in speaker_topics.items():
        possible_topics.append(value)

    return list(set(possible_topics))


def query_dataframe(df, speakername, topic):
    """
    Function that queries a dataframe based on a user-specified speaker and
    topic
    """

    return df['speechtext'][df['subtopic'] == topic][df['speakername'] == speakername]


def summarize_with_textrank(df, speakername, topic):
    """
    Function that produces a summary using the TextRank algorithm
    """

    speeches = query_dataframe(df, speakername, topic)

    # Store speeches as a concatenated string (to be passed to TextRank)
    # Store speeches as list (to be returned by this function)
    original_passages = []
    text_to_summarize = ''
    count = 0
    for index, value in speeches.items():
        original_passages.append(value)
        text_to_summarize = text_to_summarize + ' ' + value
        count += 1

    raw_summary = summarize(text_to_summarize, ratio=0.2)
    tokenized_sentences = tokenize.sent_tokenize(raw_summary)

    if len(tokenized_sentences) > num_sentences:
        summary = ' '.join(tokenized_sentences[0:num_sentences])
    else:
        summary = ' '.join(tokenized_sentences)

    print(summary)
    return original_passages, summary


def summarize_with_BERT(df, speakername, topic):
    """
    Function that produces a summary by first embedding sentences using BERT,
    clustering those embeddings, and selecting the sentence from each cluster
    that's closest to the centroid
    """

    speeches = query_dataframe(df, speakername, topic)

    # Store speeches as a concatenated string (to be split, by sentence, for
    # embedding). Also, store speeches as list (to be returned by this function)
    text_to_summarize = ''
    original_passages =[]
    for index, value in speeches.items():
        original_passages.append(value)
        text_to_summarize = text_to_summarize + ' ' + value

    # Segment concatenated string into sentences
    tokenized_sentences = tokenize.sent_tokenize(text_to_summarize)
    speech_embeddings = embedder.encode(tokenized_sentences)

    # Perform K-Means clustering. Summary will consist of sentence in each
    # cluster that is closest to the centroid
    num_clusters = num_sentences
    km = KMeans(num_clusters)
    preds = km.fit_predict(speech_embeddings)
    summary = ''
    for this_label in range(num_clusters):
        mask = preds == this_label
        test = np.linalg.norm(np.array(speech_embeddings) - km.cluster_centers_[this_label], axis=1)
        min_ind = np.argmin(test, axis=0)
        best_sentence = tokenized_sentences[min_ind]
        if best_sentence != '':
            summary += best_sentence + ' '

    return original_passages, summary
