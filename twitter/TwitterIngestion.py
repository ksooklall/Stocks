from credentials import consumer_key, consumer_secret_key, access_token, access_token_secret

import tweepy
from tweepy import API, Cursor
from datetime import datetime
from tweepy.streaming import StreamListener
from tweepy import Stream
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from peopletag import users, tags

class TwitterAuthenticator():
	"""
	Authenticator for twitter api
	"""
	def authenticate_twitter_app(self):	
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret_key)
		auth.set_access_token(access_token, access_token_secret) 
		return auth
		
class TwitterIngestion():

	def __init__(self, twitter_user=None):
		self.auth = TwitterAuthenticator().authenticate_twitter_app()
		self.twitter_client = API(self.auth)
		self.twitter_user = twitter_user
	
	def get_twitter_client_api(self):
		return self.twitter_client

	def get_user_timeline_tweets(self, num_tweets):
		tweets = []
		for tw in Cursor(self.twitter_client.user_timeline, id=self.twitter_user).items(num_tweets):
			tweets.append(tw)
		return tweets

	def get_friend_list(self, num_friends):
		friend_list = []
		for friend in Cursor(self.tiwtter_client.friends).items(num_friends):
			friend_list.append(friend)
		return friend_list

	def get_home_timeline(self, num_tweets):
		"""
		Get tweets on home page
		"""
		home_lst = []
		for tw in Cursor(self.twitter_client.home_timeline).items(num_tweets):
			home_lst.append(tw)
		return home_lst

	def tweets_to_df(self, tweets):
		cols = ['author', 'contributors', 'coordinates', 'created_at', 'destroy', 'entities', 'favorite', 'favorite_count', 'favorited', 'geo', 'id', 'id_str', 'in_reply_to_screen_name', 'in_reply_to_status_id', 'in_reply_to_status_id_str', 'in_reply_to_user_id', 'in_reply_to_user_id_str', 'is_quote_status', 'lang', 'parse', 'parse_list', 'place', 'retweet', 'retweet_count', 'retweeted', 'retweets', 'source', 'source_url', 'text', 'truncated', 'user']

		df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['tweets'])
		df['id'] = [t.id for t in tweets]		
		df['date'] = [t.created_at for t in tweets]			
		df['source'] = [t.source for t in tweets]			
		df['likes'] = [t.favorite_count for t in tweets]			
		df['retwee'] = [t.retweet_count for t in tweets]			

		return df

	def plotting(self, df):
		fig, ax = plt.subplots(1, 1, figsize=(16,4))
		df.plot(x='date', y='nlike', kind='line', label='likes', legend=True, ax=ax)
		df.plot(x='date', y='nretweet', kind='line', label='retweet', legend=True, ax=ax)

		plt.show()

class TwitterListener(StreamListener):
	"""
	Basic listener class
	"""
	def on_data(self, data):
		try:
			print(data)
		except BaseException as e:
			print(str(e))
		return True

	def on_error(self, status):
		"""
		Check for 420 message
		"""
		if status == 420:
			print('Rate limit reached')
			return False
		print(status)


class LiveStream():
	"""
	Streaming live tweets locally
	tweet_analyzer = TweetAnalyzer()

	"""
	def __init__(self):
		self.auth  = TwitterAuthenticator()

	def stream_tweets(self, fetch_tweets_filename, hash_tags):
		listener = TwitterListener()
		auth = self.auth.authenticate_twitter_app()	
		steam = Stream(auth, listener)
		steam.filter(track=hash_tags)


if __name__ == '__main__':
	#hash_tag = ['LYFT']
	#user = 'jimcramer'
	import pdb; pdb.set_trace()
	users= {'hedge_fund_manager': 'Carl_C_Icahn', 'hedge_fund_manager': 'PeterCWarren', 'hedge_fund_manag    er': 'BergenCapital', 'hedge_fund_manager': 'mark_dow', 'hedge_fund_manager': 'lexvandam', 'hedge_fun    d_manager': 'timseymour'} 
	
	df = pd.DataFrame()
	for title, user in users.items():
		twitter_client = TwitterIngestion(user)
		api = twitter_client.get_twitter_client_api()
		print('Getting {} tweets'.format(user))
		tweets = api.user_timeline(screen_name=user, count=20)
		tdf = twitter_client.tweets_to_df(tweets)	
		tdf['user'] = user
		df = pd.concat([df, tdf])

	print(df)
	import pdb; pdb.set_trace()
	#twitter = TwitterStreamer()
	#twitter.stream_tweets('', hash_tag)
