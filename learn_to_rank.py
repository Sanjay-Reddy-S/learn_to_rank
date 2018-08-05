"""
Movielens Dataset of 100K ratings: (Rated from 1 to 5)
https://grouplens.org/datasets/movielens/100k/

Referred: 
https://towardsdatascience.com/learning-to-rank-with-python-scikit-learn-327a5cfd81f
"""

import random
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import pdb
from sklearn.metrics import *
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import itertools
import csv
from datetime import datetime
import dateutil
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#List of the 19 genres present
genres_data = pd.read_csv('movielens-dataset/u.genre',sep='|',encoding = "ISO-8859-1",header = None, names = ['name','id'])
#Appends the 19 to other columns below
movie_data_cols = np.append(['movie_id','title','release_date','video_release_date','url'],genres_data['name'].values)
#Reads the movie related data into a pandas DF
movie_data = pd.read_csv('movielens-dataset/u.item',sep='|',encoding = "ISO-8859-1",header = None, names = movie_data_cols, index_col = 'movie_id')


#Prints few top rows. Contains title, release_date and genres it is in.
print "Sample movies..."
selected_cols = np.append(['title','release_date'],genres_data['name'].values)
movie_data = movie_data[selected_cols]
movie_data['release_date'] = pd.to_datetime(movie_data['release_date'])
print movie_data.head()
print

#Reading user ratings for the movie
ratings_data = pd.read_csv('movielens-dataset/u.data', sep = '\t', encoding = "ISO-8859-1", header = None, names=['user_id', 'movie_id', 'rating', 'timestamp'])
movie_data['ratings_average'] = ratings_data.groupby(['movie_id'])['rating'].mean() #Adding average rating to the movie data
movie_data['ratings_count'] = ratings_data.groupby(['movie_id'])['rating'].count() #Adding average users who rated the movie

#Prints few top rows, containing title, avg. rating and rating count
print movie_data[['title', 'ratings_average', 'ratings_count']].head()
print

#Remove movies with unknown release dates
null_release_dates = movie_data[movie_data['release_date'].isnull()]
movie_data = movie_data.drop(null_release_dates.index.values)
assert movie_data[selected_cols].isnull().any().any() == False


oldest_date = pd.to_datetime(movie_data['release_date']).min()
most_recent_date = pd.to_datetime(movie_data['release_date']).max()
normalised_age = (most_recent_date - pd.to_datetime(movie_data['release_date'])) / (most_recent_date - oldest_date)
normalised_rating = (5 - movie_data['ratings_average']) / (5 - 1)

#We will now calculate price of the movie. Range is from 0 to 10. (Use normalized age & rating)
movie_data['price'] = np.round((1 - normalised_rating) * (1 - normalised_age) * 10) #Synthetically made price now created for each movie.
#Prints few top rows, containing title, avg. rating and rating count and of course the price.
print movie_data[['title', 'price', 'ratings_average', 'ratings_count']].head()

#We will assume, customer always buys least priced movie. Our ML model should have to predict this.
movie_data = movie_data[movie_data['price'].notnull()]
movie_data['buy_probability'] = 1 - movie_data['price'] * 0.1 #Inversely proportional to the more price.

plt.plot(movie_data['price'].values, movie_data['buy_probability'].values, 'ro')
plt.xlabel('price')
plt.ylabel('buy_probability')
plt.title('Expected trend (Ground truth)')
plt.show() #Should be a line of -ve slope

class User:
	def __init__(self,id):
		self.id = id
		self.pos = [] #Movie user bought
		self.neg = [] #Movie which user saw but didn't buy

	def add_pos(self,movie_id):
		self.pos.append(movie_id)

	def add_neg(self,movie_id):
		self.neg.append(movie_id)

	def get_pos(self):
		return self.pos

	def get_neg(self):
		return self.neg

#np.random.seed(1)
class EventsGenerator:
    #Create synthetic data
    #generate 1000 users, each opens 20 movies
    def __init__(self, train_data, buy_probability):
        self.train_data = train_data
        self.buy_probability = buy_probability
        self.users = [] #objects of class User
        for id in range(1, 1000):
            self.users.append(User(id))

    def add_pos_neg(self, user, opened_movies):
        #For the opened movies, split into +ve and -ve.
        for movie_id in opened_movies:
            if np.random.binomial(1, self.buy_probability.loc[movie_id]): #binomial distribution with mean as buy_probability
                user.add_pos(movie_id)
            else:
                user.add_neg(movie_id)
                
    def build_events_data(self):
        events_data = [] #Will consist all the fields from movie_data except buy_prob [That's the answer!] and an additional field: outcome. He selected or not
        
        for user in self.users:
            for pos_id in user.get_pos():
                tmp = self.train_data.loc[pos_id].to_dict()
                tmp['outcome'] = 1
                events_data += [tmp]
            
            for neg_id in user.get_neg():
                tmp = self.train_data.loc[neg_id].to_dict()
                tmp['outcome'] = 0
                events_data += [tmp]
                
        return pd.DataFrame(events_data)
    
    def build_pairwise_events_data(self):
        events_data = [] #Contains pairwise events. pair_event_1: <customer_1, movie_1, fail, movie_3, success>
        
        for i, user in enumerate(self.users):
            print("{} of {}".format(i, len(self.users)))
            positives = user.get_pos()
            negatives = user.get_neg()
            
            sample_size = min(len(positives), len(negatives))
            
            positives = np.random.choice(positives, sample_size)
            negatives = np.random.choice(negatives, sample_size)
            
            # print("Adding {} events".format(str(len(positives) * len(negatives) * 2)))
            for positive in positives:
                for negative in negatives:                    
                    e1 = self.train_data.loc[positive].values
                    e2 = self.train_data.loc[negative].values
                    
                    pos_neg_example = np.concatenate([e1, e2, [1]])
                    neg_pos_example = np.concatenate([e2, e1, [0]])
                    
                    events_data.append(pos_neg_example)
                    events_data.append(neg_pos_example)
        
        c1 = [ c + '_1' for c in train_data.columns]
        c2 = [ c + '_2' for c in train_data.columns]
        return pd.DataFrame(events_data, columns = np.concatenate([c1, c2, ['outcome']]))

    def generate(self, pairwise=False):
        #Generates the events and returns as a pd DF
        for user in self.users:
            opened_movies = np.random.choice(self.train_data.index.values, 20) #Select 20 movie_ids randomly
            self.add_pos_neg(user, opened_movies)
        if pairwise:
            return self.build_pairwise_events_data()
        else:
            return self.build_events_data()

def build_train_data(movie_data): #For movie_data Df, extract normalized features and remove buy_prob. from it.
	feature_cols = np.setdiff1d(movie_data.columns, np.array(['title','buy_probability']))
	train_data = movie_data.loc[:,feature_cols]

	scaler = StandardScaler() #For normalizing. ALso centred around zero. Including price
	train_data.loc[:, ('price')] = scaler.fit_transform(train_data[['price']])
	train_data[('ratings_average')] = scaler.fit_transform(train_data[['ratings_average']])
	train_data[('ratings_count')] = scaler.fit_transform(train_data[['ratings_count']])
	train_data[('release_date')] = train_data['release_date'].apply(lambda x: x.year) #Only want year
	train_data[('release_date')] = scaler.fit_transform(train_data[['release_date']])

	return train_data

def get_fet_cols(train_data, pairwise=False):
    # If not pairwise, simply return the train_data column values
    if not pairwise:
        return train_data.columns.values
    else:
        f1 = [c + '_1' for c in train_data.columns.values]
        f2 = [c + '_2' for c in train_data.columns.values]
        f1.extend(f2)
        return np.asarray(f1)
"""
def save_events_data(events_data, train_data, tag, pairwise=False):
    events_data = events_data.reindex(np.random.permutation(events_data.index))
    events_data.to_csv('movie_events_' + tag + '.csv')
    
    if not pairwise:
        df = pd.DataFrame(get_fet_cols(train_data))
        df.to_csv("feature_columns_" + tag + ".csv")
    else:
        df = pd.DataFrame(get_fet_cols(train_data, pairwise=True))
        df.to_csv("feature_columns_" + tag + ".csv")

def load_events_data(tag):
    events_data = pd.DataFrame.from_csv('movie_events_' + tag + '.csv')
    tmp = pd.DataFrame.from_csv("feature_columns_" + tag + ".csv")
    feature_columns = tmp['0'].values
    
    return [events_data, feature_columns]
"""

def get_test_train_data(events_data, feature_columns):
    # Using train_test_split, create train & test sets
    X = events_data.loc[:, feature_columns].values.astype(np.float32)
    print('overall input shape: ' + str(X.shape))

    y = events_data.loc[:, ['outcome']].values.astype(np.float32).ravel()
    print('overall output shape: ' + str(y.shape))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('training input shape: ' + str(X_train.shape))
    print('training output shape: ' + str(y_train.shape))

    print('testing input shape: ' + str(X_test.shape))
    print('testing output shape: ' + str(y_test.shape))
    
    return [X_train, X_test, y_train, y_test]

def plot_events_distribution(events_data):
    # Plot distribution of events data
    events_data_sample = events_data.sample(frac=0.1)
    negative_outcomes = events_data_sample[events_data_sample['outcome'] == 0.0]['price']
    positive_outcomes = events_data_sample[events_data_sample['outcome'] == 1.0]['price']
    
    outcomes = np.array(list(zip(negative_outcomes.values, positive_outcomes.values)))
    plt.hist(outcomes, bins=11, label = ['Negative', 'Positive'])
    plt.legend()
    plt.xlabel('price')
    plt.show()

def plot_rank(features, model, train_data, predict_fun,title=None):
    # Plot prediction, vs true	
    lg_input = train_data.values.astype(np.float32)
    print('overall input shape: ' + str(lg_input.shape))

    train_data_with_rank = train_data.copy()
    train_data_with_rank['rank'] = predict_fun(model, lg_input)
    
    for idx, feature in enumerate(features):
        plt.subplot(len(features), 1, idx + 1)
        plt.plot(train_data_with_rank[feature].values, train_data_with_rank['rank'].values, 'ro')
        plt.xlabel(feature)
        plt.ylabel('rank')
    plt.plot([-3,-2,-1,0,1,2,3],[1.0,0.83,0.66,0.5,0.33,0.17,0.0],'b')    
    plt.tight_layout()
    if title!=None:
        plt.title(title)
    plt.show()

def train_model(model, prediction_function, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_train_pred = prediction_function(model, X_train)

    print('train precision: ' + str(precision_score(y_train, y_train_pred)))
    print('train recall: ' + str(recall_score(y_train, y_train_pred)))
    print('train accuracy: ' + str(accuracy_score(y_train, y_train_pred)))

    y_test_pred = prediction_function(model, X_test)

    print('test precision: ' + str(precision_score(y_test, y_test_pred)))
    print('test recall: ' + str(recall_score(y_test, y_test_pred)))
    print('test accuracy: ' + str(accuracy_score(y_test, y_test_pred)))
    
    return model

def get_predicted_outcome(model, data):
    return np.argmax(model.predict_proba(data), axis=1).astype(np.float32)

def get_predicted_rank(model, data):
    return model.predict_proba(data)[:, 1]

class PerfectPredictor:
    def fit(self, X, y):
        return None
    
    def predict(self, X):
        min_max_scaler = preprocessing.MinMaxScaler()
        return 1 - min_max_scaler.fit_transform(X[:, -5])

train_data = build_train_data(movie_data) #Removes buy_prob from movie data
events_data = EventsGenerator(train_data, movie_data['buy_probability']).generate()
feature_cols = get_fet_cols(train_data)
#save_events_data(events_data, train_data, 'linear')
#events_data, feature_columns = load_events_data('linear')
plot_events_distribution(events_data)

X_train, X_test, y_train, y_test = get_test_train_data(events_data, feature_cols)
model = train_model(LogisticRegression(), get_predicted_outcome, X_train, y_train, X_test, y_test)
plot_rank(['price'], model, train_data, get_predicted_rank,title="Prediction of rank vs price...")