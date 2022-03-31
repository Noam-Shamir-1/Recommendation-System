import abc
from typing import Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import math


class Recommender(abc.ABC):
    def __init__(self, ratings: pd.DataFrame):
        self.initialize_predictor(ratings)
        self.users = set(ratings.user.values)

    @abc.abstractmethod
    def initialize_predictor(self, ratings: pd.DataFrame):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        raise NotImplementedError()

    def rmse(self, true_ratings) -> float:
        """
        :param true_ratings: DataFrame of the real ratings
        :return: RMSE score
        """
        def predict(row):
            return self.predict(row["user"], row["item"], row["timestamp"])

        y_actual = true_ratings["rating"].tolist()
        true_ratings["p"] = true_ratings.apply(lambda row: predict(row), axis=1)
        y_predicted = true_ratings["p"].tolist()
        sum = 0
        for i in range(len(y_actual)):
            sum += (y_actual[i]-y_predicted[i])**2
        return math.sqrt((1/len(y_actual))*sum)


class BaselineRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.r_tilda = {}
        self.ratings = ratings
        self.r_avg = ratings['rating'].mean()
        self.b_u = {}
        u_mean = ratings.groupby('user')['rating'].mean()
        for user in u_mean.index:
            self.b_u[user] = u_mean[user] - self.r_avg

        self.b_i = {}
        i_mean = ratings.groupby('item')['rating'].mean()
        for item in i_mean.index:
            self.b_i[item] = i_mean[item] - self.r_avg

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        x = self.r_avg + self.b_u[user] + self.b_i[item]
        if x > 5:
            return 5
        elif x < 0.5:
            return 0.5
        else:
            return x

class NeighborhoodRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.users = set(ratings.user.values)
        self.ratings = ratings
        self.r_avg = ratings['rating'].mean()
        self.b_u = {}

        self.ratings["r_tilda"] = self.ratings.apply(lambda row: row.rating - self.r_avg, axis=1)

        u_mean = ratings.groupby('user')['rating'].mean()
        for user in u_mean.index:
            self.b_u[user] = u_mean[user] - self.r_avg

        self.b_i = {}
        i_mean = ratings.groupby('item')['rating'].mean()
        for item in i_mean.index:
            self.b_i[item] = i_mean[item] - self.r_avg



        self.user_rating = {u: {} for u in self.users}
        for index, row in self.ratings.iterrows():
            self.user_rating[row["user"]][row["item"]] = row.r_tilda

        self.user_similar = {u: [] for u in self.users}
        for user1 in self.users:
            all_sim_for_user = []
            for user2 in self.users:
                if user1 != user2:
                    corr = self.user_similarity(user1, user2)
                    all_sim_for_user.append((abs(corr), corr, user2))
            self.user_similar[user1] = sorted(all_sim_for_user, reverse=True)

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """

        # find 3 closest neighbors:
        sum_corr_neighbors_r_tilda = 0
        sum_corr_neighbors_abs = 0
        counter = 0
        for neighbor in self.user_similar[user]:
            if item in self.user_rating[neighbor[2]]:
                sum_corr_neighbors_r_tilda += neighbor[1] * self.user_rating[neighbor[2]][item]
                sum_corr_neighbors_abs += neighbor[0]
                counter += 1
            if counter >= 3:
                break

        if sum_corr_neighbors_abs != 0:
            x = self.r_avg + self.b_u[user] + self.b_i[item] + sum_corr_neighbors_r_tilda/sum_corr_neighbors_abs
        else:
            x = self.r_avg + self.b_u[user] + self.b_i[item]

        if x > 5:
            return 5
        elif x < 0.5:
            return 0.5
        else:
            return x

    def user_similarity(self, user1: int, user2: int) -> float:
        """
        :param user1: User identifier
        :param user2: User identifier
        :return: The correlation of the two users (between -1 and 1)
        """
        item1 = set(self.user_rating[user1].keys())
        item2 = set(self.user_rating[user2].keys())
        item_both = item1.intersection(item2)
        if len(item_both) == 0:
            return 0

        vec1 = []
        vec2 = []
        for i in item_both:
            vec1.append(self.user_rating[user1][i])
            vec2.append(self.user_rating[user2][i])
        np_vec1 = np.array(vec1)
        np_vec2 = np.array(vec2)
        # return of similarity
        return np.dot(np_vec1.T, np_vec2) / (np.linalg.norm(np_vec1) * np.linalg.norm(np_vec2))


class LSRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.rating = ratings
        self.b = 0
        self.r_avg = ratings['rating'].mean()
        rating = self.rating.sort_values(by='user', ascending=True)
        self.y = self.rating['rating'].tolist() - self.r_avg
        users = self.rating['user'].unique().tolist()
        users.sort()
        items = self.rating['item'].unique().tolist()
        items.sort()

        self.x = []
        len_vector_b = len(users) + len(items) + 3

        for index, row in self.rating.iterrows():
            user = row["user"]
            item = row["item"]
            date_time = row["timestamp"]
            time = datetime.fromtimestamp(date_time)
            self.x.append(np.zeros(len_vector_b))

            today_at_hour_6 = datetime(time.year, time.month, time.day, 6, 0, 0, 0)
            today_at_hour_18 = datetime(time.year, time.month, time.day, 18, 0, 0, 0)

            self.x[int(index)][int(user)] = 1
            self.x[int(index)][len(users)+int(item)] = 1


            if time.weekday() in [4,5]:
                self.x[int(index)][len_vector_b - 1] = 1
            else:
                self.x[int(index)][len_vector_b - 1] = 0

            if today_at_hour_6 <= time and time <= today_at_hour_18:
                self.x[int(index)][len_vector_b - 2] = 1
            else:
                self.x[int(index)][len_vector_b - 3] = 1


    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """

        users = self.rating['user'].unique().tolist()

        time = datetime.fromtimestamp(timestamp)
        if time.weekday() in [4, 5]:
            bw = self.b.item(len(self.b)-1)
        else:
            bw = 0
        today_at_hour_6 = datetime(time.year, time.month, time.day, 6, 0, 0, 0)
        today_at_hour_18 = datetime(time.year, time.month, time.day, 18, 0, 0, 0)
        if time > today_at_hour_6 and time < today_at_hour_18:
            b_n_d = self.b.item(len(self.b)-2)
        else:
            b_n_d = self.b.item(len(self.b)-3)

        x = self.r_avg + self.b.item(int(user)) + self.b.item(len(users)+int(item)) + b_n_d + bw

        if x > 5:
            return 5
        elif x < 0.5:
            return 0.5
        else:
            return x

    def solve_ls(self)-> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Creates and solves the least squares regression
        :return: Tuple of X, b, y such that b is the solution to min ||Xb-y||
        """

        self.b = np.linalg.lstsq(self.x, self.y, rcond=None)[0]
        return self.x, self.b, self.y

class CompetitionRecommender(Recommender):
    def initialize_predictor(self, ratings: pd.DataFrame):
        self.r_tilda = {}
        self.ratings = ratings
        self.r_avg = ratings['rating'].mean()

        self.b_u = {}
        u_mean = ratings.groupby('user')['rating'].mean()
        for user in u_mean.index:
            self.b_u[user] = u_mean[user] - self.r_avg

        self.b_i = {}
        i_mean = ratings.groupby('item')['rating'].mean()
        for item in i_mean.index:
            self.b_i[item] = i_mean[item] - self.r_avg

    def predict(self, user: int, item: int, timestamp: int) -> float:
        """
        :param user: User identifier
        :param item: Item identifier
        :param timestamp: Rating timestamp
        :return: Predicted rating of the user for the item
        """
        x = self.r_avg + self.b_u[user] + self.b_i[item]
        if x > 5:
            return 5
        elif x < 0.5:
            return 0.5
        else:
            return x
