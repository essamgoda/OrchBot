import os
from flask import Flask, request
from fbmessenger import BaseMessenger
from fbmessenger import quick_replies
from fbmessenger.elements import Text
from fbmessenger.thread_settings import GreetingText, GetStartedButton, MessengerProfile, PersistentMenu, PersistentMenuItem

def process_message(message):
    app.logger.debug('Message received: {}'.format(message))

    if 'attachments' in message['message']:
        if message['message']['attachments'][0]['type'] == 'location':
            app.logger.debug('Location received')
            latitude, longitude = message['message']['attachments'][0]['payload']['coordinates']['lat'], message['message']['attachments'][0]['payload']['coordinates']['long']
            response = Text(text='Received Latitude: {} Longitude: {}'.format(latitude, longitude))
            return response.to_dict()

class Messenger(BaseMessenger):
    def __init__(self, page_access_token):
        self.page_access_token = page_access_token
        super(Messenger, self).__init__(self.page_access_token)

    def message(self, message):
        action = process_message(message)
        res = self.send(action,'RESPONSE')
        app.logger.debug('Response: {}'.format(res))

    def delivery(self, message):
        pass

    def read(self, message):
        pass

    def account_linking(self, message):
        pass

    def postback(self, message):
        payload = message['postback']['payload']

        if 'start' in payload:
            quick_reply_1 = quick_replies.QuickReply(title='Location', content_type='location')
            quick_replies_set = quick_replies.QuickReplies(quick_replies=[
                quick_reply_1
            ])
            text = {'text': 'Share your location'}
            text['quick_replies'] = quick_replies_set.to_dict()
            self.send(text,'RESPONSE')

    def optin(self, message):
        pass

    def init_bot(self):
        greeting_text = GreetingText('Welcome to weather bot')
        get_started = GetStartedButton(payload='start')

        menu_item_1 = PersistentMenuItem(item_type='postback', title='Weather Status', payload='start')
        menu = PersistentMenu(menu_items=[menu_item_1])

        messenger_profile = MessengerProfile(persistent_menus=[menu], get_started=get_started, greetings=[greeting_text])
        messenger.set_messenger_profile(messenger_profile.to_dict())


app = Flask(__name__)

# token to verify that this bot is legit
verify_token = os.getenv('VERIFY_TOKEN', None)
# token to send messages through facebook messenger
access_token = os.getenv('ACCESS_TOKEN', None)

app.debug = True
messenger = Messenger(os.environ.get('ACCESS_TOKEN'))


@app.route('/webhook', methods=['GET', 'POST'])
def webhook():
    if request.method == 'GET':
        if request.args.get('hub.verify_token') == verify_token:
            messenger.init_bot()
            return request.args.get('hub.challenge')
        raise ValueError('FB_VERIFY_TOKEN does not match.')
    elif request.method == 'POST':
        messenger.handle(request.get_json(force=True))
    return ''


if __name__ == '__main__':
    app.run(host='0.0.0.0')
