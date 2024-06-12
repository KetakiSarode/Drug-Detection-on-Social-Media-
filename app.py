from flask import Flask, render_template, request, jsonify, session
import instaloader
import csv
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text
import pandas as pd
from cleantext import clean
import numpy as np
import requests

model = tf.keras.models.load_model('C:/Users/lynes/bert model/bert_model')

app = Flask(__name__)

global_username = None

def search_top_profiles_with_hashtag(hashtag):
    # Create an instance of Instaloader class
    L = instaloader.Instaloader()
    L.login('lyneshiaacorrea','lyneshiarules123$')

    # Initialize an empty list to store profile usernames
    top_profiles = []

    try:
        # Search for the hashtag and retrieve top profiles
        search_results = instaloader.TopSearchResults(L.context, hashtag)

        # Iterate over the top profiles
        for profile in search_results.get_profiles():
            # Get the username of the profile
            username = profile.username
            # Add the username to the list if it's not already present
            if username not in top_profiles:
                top_profiles.append(username)
            # Break if we have found top 10 profiles
            if len(top_profiles) == 10:
                break
    except Exception as e:
        print("Error occurred:", e)

    return top_profiles

def download_posts_and_captions(username):
    # Create an instance of Instaloader class
    L = instaloader.Instaloader()
    L.login('lyneshiaacorrea','lyneshiarules123$')

    # Get profile instance
    profile = instaloader.Profile.from_username(L.context, username)

    # Get profile photo, username, and bio
    profile_photo = profile.profile_pic_url
    username = profile.username
    bio = profile.biography

    # Specify the path where you want to save the CSV file
    csv_file_path = f'posts_and_captions.csv'

    # Create a CSV file to store posts and captions
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Post URL', 'Caption'])

        # Iterate over the posts and save top 10 posts and captions into CSV file
        count = 0
        for post in profile.get_posts():
            csv_writer.writerow([post.url, post.caption])
            count += 1
            if count == 10:  # Save top 10 posts
                break

    return csv_file_path, profile_photo, username, bio

def results():
    df = pd.read_csv("posts_and_captions.csv")
    df_results = df['Caption']
    df_results_cleaned = df_results.apply(lambda x: clean(x, no_emoji=True))
    captions = model.predict(df_results_cleaned)
    res = np.where(captions > 0.5, 'Yes', 'No')
    res = res.flatten().tolist()
    return zip(df_results, res)

@app.route("/", methods=["GET"])
def home():
    return render_template('home.html')

@app.route("/frontpage", methods=["GET"])
def front():
    return render_template('frontpage.html')

@app.route("/hashtag", methods=["GET"])
def hashtag():
    return render_template('/Hashtag/hashtag.html')

@app.route('/search_keywords', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        hashtag = request.form['hashtag']
        top_profiles = search_top_profiles_with_hashtag(hashtag)
        return render_template('Hashtag/hashtag_profiles.html', top_profiles=top_profiles)
    return render_template('hashtag.html')

@app.route('/get_profile_info', methods=['GET'])
def profile():
    return render_template('Profile/profile.html')

@app.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        global global_username
        username = request.form['username']
        global_username = username
        csv_file_path, profile_photo, username, bio = download_posts_and_captions(username)

        # Read CSV file and pass data to template
        with open(csv_file_path, 'r', newline='', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            # Skip header row
            next(csv_reader)
            csv_data = list(csv_reader)

        return render_template('Profile/table.html', csv_data=csv_data, profile_photo=profile_photo, username=username, bio=bio)

@app.route('/results', methods=['GET'])
def search_for_drugs():
    bert_model_results = results()
    # Assume 'username' is defined here or retrieve it from the session or request
    return render_template('Model Results/results.html', bert_model_results=bert_model_results)

@app.route('/report', methods=['GET'])
def report():

    # If username is not in the session, try to get it from the request args
    global global_username
    username = global_username
    return render_template('report.html', username=username)


if __name__ == '__main__':
    app.run(debug=True)
