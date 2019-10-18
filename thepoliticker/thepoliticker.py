# -*- coding: utf-8 -*-

from flask import Flask, render_template, request, session
import pandas as pd
import json
import os
import sys
sys.path.append('./thepoliticker')
import summarization as summ

# Create the application object
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load the dataframe
df = summ.load_dataframe('../data/cleaned/hansard_cleaned.pkl')

# Get fixed information for server session
possible_speakers = summ.get_possible_speakers(df)
speaker_pic_urls = summ.get_img_urls('https://www.ourcommons.ca/Parliamentarians/en/members?page=')

# Function to check availability of URL to a selected speaker's picture
def check_pic_availability(speaker):
	if speaker in speaker_pic_urls.keys():
		pic_url = speaker_pic_urls[speaker]
	else:
		pic_url = '../static/img/placeholder.png'

	return pic_url

# Define default page routing
@app.route('/',methods=["GET","POST"])
def home_page():

	#session['possible_speakers'] = '###'.join(get_possible_speakers(df))
	return render_template('index.html',
						possible_speakers=possible_speakers)  # render a template

# Process speaker
@app.route('/process_speaker')
def process_speaker():

	# Pull user input from form
	session['speaker_input'] = request.args.get('speaker_input')

	possible_topics = summ.get_speaker_topics(df, session['speaker_input'])

	pic_url = check_pic_availability(session['speaker_input'])

	return render_template("index.html",
						possible_speakers=possible_speakers,
						speaker_input=session['speaker_input'],
						possible_topics=possible_topics,
						pic_url= pic_url,
	                    my_form_status="speaker_only")

# Define routing once user provides topic
@app.route('/output')
def process_topic():

	possible_topics = summ.get_speaker_topics(df, session['speaker_input'])

	# Pull user input from form
	topic_input = request.args.get('topic_input')

   	# Apply summarization algorithm(s)
	original_passages, speech_dates, summary = summ.summarize_speeches(df, session['speaker_input'], topic_input, 'TextRank')
	#original_passages, summary = summ.summarize_speeches(df, session['speaker_input'], topic_input, 'BERT')

   	# Case if empty
	if session['speaker_input'] != '' and (topic_input is None or topic_input == ''):
		# Render original page
		return render_template("index.html",
							possible_speakers=possible_speakers,
							speaker_input=session['speaker_input'],
							possible_topics=possible_topics,
							my_form_status="speaker_only")
	else:

		pic_url = check_pic_availability(session['speaker_input'])

		# Render page with result of user's query
		return render_template("index.html",
					possible_speakers=possible_speakers,
   					speaker_input=session['speaker_input'],
					topic_input=topic_input,
                    summary=summary,
					original_passages=original_passages,
					speech_dates=speech_dates,
					pic_url= pic_url,
                 	my_form_status="complete")

# Start the server with the 'run()' method
if __name__ == "__main__":
	app.run(debug=True) # Will run locally http://127.0.0.1:5000/
