from flask import Flask,render_template, request, Response,send_file,abort, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
import requests, json, random, os,zipfile
import re
import numpy as np
import pandas as pd
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
import numpy as np
import tflearn
import tensorflow as tf

from os import listdir

import csv,datetime

import io
from PIL import Image
from ObjectDetector import Detector
import urllib.request

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./db/database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
db.init_app(app)

import Models
#from Model import create_user,create_msg

# DATABASE = "./db/database.db"

# env_variables
quick_replies_list = [{
    "content_type":"text",
    "title":"QA",
    "payload":"QA",
},
{
    "content_type":"text",
    "title":"Software devloper",
    "payload":"Software devloper",
},
{
    "content_type":"text",
    "title":"Web Developer",
    "payload":"Web Developer",
},
{
    "content_type":"text",
    "title":"Designer",
    "payload":"Design",
}
]

questions=['Please enter your E-mail','Please enter your mobile number','Please Tell me more about yourself..', 'What sparked your interest in web development?','What are the responsibilities of a web developer?','What programming languages have you used in the past? ','What are your top two programming languages?','Explain the purpose of each of the HTTP request types when used with a RESTful web service?']
# token to verify that this bot is legit
verify_token = os.getenv('VERIFY_TOKEN', 'TESTINGTOKEN')
# token to send messages through facebook messenger
access_token = os.getenv('ACCESS_TOKEN', 'EAAFATk7mVvQBAKTZCS0mIpd7woSCwX9XVZBGzHsIgHt1B1a6MGVXiLU5ZCXOMLUgvSQgbmdcZClkZBzSXRVLtCJhe0P4uU7kmNk55v4ZBn7u8govLneBO2mNQCJ1kojVm4I7b9o8Ux9vYu7EMFEBMIauOfLrZBh0kqwFufU4aeoZCNmDz6kSvwu3')

with open('./intents1.json') as json_data:
    intents = json.load(json_data)

#ml
words = []
classes = []
documents = []
ignore_words = ['?']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

#print (len(documents), "documents")
#print (len(classes), "classes", classes)
#print (len(words), "unique stemmed words", words)

# create our training data
training = []
output = []
# create an empty array for our output
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)

train_x = list(training[:,0])
train_y = list(training[:,1])

# reset underlying graph data
tf.reset_default_graph()
# Build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)
# Define model and setup tensorboard

model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')
# load our saved model
model.load('./model.tflearn')



def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))



ERROR_THRESHOLD = 0.25
def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bow(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # a random response from the intent
                    return random.choice(i['responses'])

            results.pop(0)



#ml

@app.route('/webhook', methods=['GET'])
def webhook_verify():
    if request.args.get('hub.verify_token') == verify_token:
        return request.args.get('hub.challenge')
    return "Wrong verify token"

@app.route('/webhook', methods=['POST'])
def webhook_action():
    data = json.loads(request.data.decode('utf-8'))
    #import pdb; pdb.set_trace()

    for entry in data['entry']:
        messaging = entry['messaging']
        for message in messaging:
            if message.get('message'):

                user_id = message['sender']['id']
                user_details_url = "https://graph.facebook.com/v2.6/%s"%user_id
                user_details_params = {'fields':'first_name,last_name,profile_pic', 'access_token':access_token}
                user_details = requests.get(user_details_url, user_details_params).json()
                user_name=user_details['first_name']
                userid=int(user_id)

                Models.create_user(user_name,0,userid)
                count=Models.get_counter(userid)

                if message['message'].get('text'):
                    msg=message['message']['text']

                    if count==len(questions):
                            Models.create_msg(msg,userid)
                            count=0
                            Models.update_counter(count,userid)
                            response = {
                                'recipient': {'id': user_id},
                                'message': {}
                            }
                            response['message']['text'] = "Thanks for your time"
                    elif count!=0:
                            Models.create_msg(msg,userid)
                            response = {
                                'recipient': {'id': user_id},
                                #'message': {"text": 'We are looking for an experienced WordPress and open source CMS developer to join our friendly and hard-working Website team, %s!' % link}
                                'message': {"text": questions[count]}
                            }
                            count+=1
                            Models.update_counter(count,userid)
                    elif 'vacancy'in msg or 'vacancies' in msg:
                        response = {
                            'recipient': {'id': user_id},
                            'message': {"text": 'These are the available vacancies ',"quick_replies":quick_replies_list}
                        }
                    elif 'Web Developer' in msg:
                        response = {
                            'recipient': {'id': user_id},
                            #'message': {"text": 'We are looking for an experienced WordPress and open source CMS developer to join our friendly and hard-working Website team, %s!' % link}
                            'message': {"text": questions[count]}
                        }
                        count+=1
                        Models.update_counter(count,userid)

                    elif 'hi' in msg.lower():
                        response = {
                            'recipient': {'id': user_id},
                            'message': {"text": 'Hi %s!' % user_name}
                        }

                        Models.update_counter(0,userid)

                    else:
                        user_message = entry['messaging'][0]['message']['text']
                        response = {
                            'recipient': {'id': user_id},
                            'message': {}
                        }
                        response['message']['text'] = handle_message(user_id, user_message)

                elif message['message'].get('attachments'):
                    attachment_link = message["message"]["attachments"][0]["payload"]["url"]
                    print(attachment_link)

                    fd = urllib.request.urlopen(attachment_link)
                    file = Image.open(io.BytesIO(fd.read()))

                    try:

                        detector = Detector()
                        img,label = detector.detectObject(file)

                        response = {
                            'recipient': {'id': user_id},
                            'message': {}
                        }
                        response['message']['text'] = label
                    except:
                        response = {
                            'recipient': {'id': user_id},
                            'message': {}
                        }
                        response['message']['text'] = "couldn't catch that..."
                else:
                    response = {
                        'recipient': {'id': user_id},
                        'message': {}
                    }
                    response['message']['text'] = "couldn't catch that..."
                params  = {"access_token": access_token}
                headers = {"Content-Type": "application/json"}

                if count!=0:
                    Models.create_msg(response['message']['text'],userid)

                r = requests.post('https://graph.facebook.com/v2.6/me/messages', params=params, headers=headers, json=response)


    return Response(response="EVENT RECEIVED",status=200)

@app.route('/webhook_dev', methods=['POST'])
def webhook_dev():
    # custom route for local development
    data = json.loads(request.data.decode('utf-8'))
    user_message = data['entry'][0]['messaging'][0]['message']['text']
    user_id = data['entry'][0]['messaging'][0]['sender']['id']
    response = {
        'recipient': {'id': user_id},
        'message': {'text': handle_message(user_id, user_message)}
    }
    return Response(
        response=json.dumps(response),
        status=200,
        mimetype='application/json'
    )

def handle_message(user_id, user_message):
    # DO SOMETHING with the user_message ... ¯\_(ツ)_/¯
    #return "Hello "+user_id+" ! You just sent me : " + user_message + "\n" + response(user_message)
    return " " + response(user_message)

def create_csv(msgs,user_name):
    df = pd.DataFrame(msgs,columns=[''])
    df.to_csv(user_name+'.csv', sep='\t', encoding='utf-8')
    # with open(user_name+'.csv', 'a') as csvFile:
    #     writer = csv.writer(csvFile, quoting=csv.QUOTE_ALL)
    #     writer.writerow(msgs)
    # csvFile.close()

@app.route('/privacy', methods=['GET'])
def privacy():
    # needed route if you need to make your bot public
    return "This facebook messenger bot's only purpose is to [...]. That's all. We don't use it in any other way."

@app.route('/', methods=['GET'])
def index():
    return "Hello there, I'm a facebook messenger bot."

#@app.route('/download_all', methods=['GET']) # this is a job for GET, not POST
def download_all():
    zipf = zipfile.ZipFile('CSV.zip','w', zipfile.ZIP_DEFLATED)
    filenames = find_csv_filenames('./')
    for name in filenames:
        zipf.write(name)
    zipf.close()
    '''return send_file('CSV.zip',
            mimetype = 'zip',
            attachment_filename= 'CSV.zip',
            as_attachment = True)'''

@app.route("/files")
def list_files():
    """Endpoint to list files on the server."""
    files = []
    for filename in os.listdir('./'):
        path = os.path.join('./', filename)
        if os.path.isfile(path):
            files.append(filename)
    return jsonify(files)

@app.route("/files/<path:path>")
def get_file(path):
    """Download a file."""
    return send_from_directory('./', path, as_attachment=True)

@app.route("/users", methods=['GET'])
def users():
    users = Models.User.query.all()
    return render_template('users.html', users=users)

@app.route("/users/<path:user_id>")
def get_users(user_id):
    """Download a file."""
    msgs=Models.get_user_msg(user_id=user_id)
    username=Models.get_user_name(user_id)
    create_csv(msgs,username)
    return send_file(username+'.csv',
            mimetype = 'csv',
            attachment_filename= username+'.csv',
            as_attachment = True)
    # return "hi "+str(user_id)


def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
