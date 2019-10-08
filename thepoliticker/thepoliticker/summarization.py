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
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt


max_sentences = 4

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

    # Two different queries to avoid issues with boolean indexing
    speeches = df['speechtext'][df['subtopic'] == topic][df['speakername'] == speakername]
    dates = df['speechdate'][df['subtopic'] == topic][df['speakername'] == speakername]

    return speeches, dates

def run_textrank(text_to_summarize, num_sentences = max_sentences):
    """
    Function that produces a summary using the TextRank algorithm
    """

    raw_summary = summarize(text_to_summarize, ratio=0.25)
    tokenized_sentences = tokenize.sent_tokenize(raw_summary)
    print(type(tokenized_sentences))

    if len(tokenized_sentences) > num_sentences:

        #summary = ' '.join(tokenized_sentences[0:num_sentences])
        summary = tokenized_sentences[0:num_sentences]
    else:
        #summary = ' '.join(tokenized_sentences)
        summary = tokenized_sentences

    return summary, num_sentences;


def run_BERT(text_to_summarize, max_sentences = max_sentences):
    """
    Function that produces a summary by first embedding sentences using BERT,
    clustering those embeddings, and selecting the sentence from each cluster
    that's closest to the centroid
    """
    # Segment concatenated string into sentences
    tokenized_sentences = tokenize.sent_tokenize(text_to_summarize)
    print(len(tokenized_sentences))
    speech_embeddings = np.array(embedder.encode(tokenized_sentences))
    print(speech_embeddings.shape)

    # Perform K-Means clustering. Summary will consist of sentence in each
    # cluster that is closest to the centroid
    #num_clusters = max_sentences
    #km = KMeans(num_clusters)
    #preds = km.fit_predict(speech_embeddings)

    distortions = []
    print(max_sentences)
    K = range(1, max_sentences+1)
    for k in K:
        km = KMeans(n_clusters=k).fit(speech_embeddings)
        km.fit(speech_embeddings)
        distortions.append(sum(np.min(cdist(speech_embeddings, km.cluster_centers_, 'euclidean'), axis=1)) / speech_embeddings.shape[0])

    # Compute first and second numerical derivatives
    distortion_prime = [a - b for a, b in zip(distortions[1:], distortions[0:-1])]
    distortion_prime.insert(0, 0)

    distortion_double_prime = [a - b for a, b in zip(distortion_prime[1:], distortion_prime[0:-1])]
    distortion_double_prime.insert(0, 0)

    plt.subplot(3,1,1)
    plt.plot(K, distortion_prime, 'bx-')
    plt.xlabel('k')
    plt.ylabel('distortion_prime')
    plt.title('distortion_prime')
    plt.show()

    # Plot the elbow
    plt.subplot(3,1,2)
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

    plt.subplot(3,1,3)
    plt.plot(K, distortion_double_prime, 'bx-')
    plt.xlabel('k')
    plt.ylabel('distortion_double_prime')
    plt.title('distortion_double_prime')
    plt.show()

    # Identify elbow/ inflection using second derivative
    num_clusters = np.argmax(np.array(distortion_double_prime), axis=0)

    # Cluster using empirically determined number of clusters
    km = KMeans(int(num_clusters))
    preds = km.fit_predict(speech_embeddings)

    summary = ''
    for this_label in range(num_clusters):
        mask = preds == this_label
        test = np.linalg.norm(np.array(speech_embeddings) - km.cluster_centers_[this_label], axis=1)
        min_ind = np.argmin(test, axis=0)
        best_sentence = tokenized_sentences[min_ind]
        if best_sentence != '':
            summary += best_sentence + ' '

    return summary, num_clusters


def summarize_speeches(df, speakername, topic, method='TextRank'):
    """
    Function that collects an MP's speeches on a given topic then passes the
    text to a summarization algorithm
    """

    speeches, dates = query_dataframe(df, speakername, topic)
    speeches = speeches.iloc[::-1]
    dates = dates.iloc[::-1]

    # Store speeches as a concatenated string (to be passed to TextRank)
    # Store speeches as list (to be returned by this function)
    original_passages = []
    original_dates = []
    text_to_summarize = ''
    count = 0
    for index, value in speeches.items():
        original_passages.append(value)
        text_to_summarize = text_to_summarize + ' ' + value
        count += 1

    for index, value in dates.items():
        original_dates.append(value)

    summary = ''
    if method == 'TextRank':
        summary = run_textrank(text_to_summarize)

    elif method == "BERT":
        summary = run_BERT(text_to_summarize)
    else:
        print('Unknown method for summarization, \'%s\'' % method)

    return original_passages, original_dates, summary
