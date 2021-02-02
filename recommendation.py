from pyspark.sql import SparkSession
from pyspark.sql.types import StringType
from pyspark.ml.tuning import CrossValidatorModel
import pyspark.sql.functions as fn
import sys
import random

import config

spark = SparkSession.builder.appName("Recommendations").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")


ratings_filepath = 'data/input/u.data'
movies_filepath = 'data/input/u.item'

ratings = spark.read.option("delimiter", "\t").csv(ratings_filepath, schema=config.rating_schema)

movies = spark.read.option("delimiter", "|").csv(movies_filepath, schema=config.movie_schema)

# load saved model and test dataset
model = CrossValidatorModel.load('als_model')
test = spark.read.parquet('data/tmp/test')

#Extract best model from the cv model above
best_model = model.bestModel
print('Best model: {}'.format(best_model))

test_predictions = best_model.transform(test)
RMSE = model.getEvaluator().evaluate(test_predictions)
print(RMSE)

# Generate 'n' Recommendations for all users
recommendations = best_model.recommendForAllUsers(config.nreco)


nrecommendations = recommendations\
	.withColumn("rec_exp", fn.explode("recommendations"))\
	.select('userId', fn.col("rec_exp.movieId"), fn.col("rec_exp.rating"))

# save the recommendation dataframe
nrecommendations.write.mode("overwrite").parquet('data/output/{}recommendations'.format(config.nreco))


### TESTING

def argmax(cols, *args):
	return [c for c, v in zip(cols, args) if v == max(args)]


def testing(user):
	argmax_udf = lambda cols: fn.udf(lambda *args: argmax(cols, *args), StringType())
	cols = ['unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', \
		'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
	movies1 = movies.withColumn("genres", argmax_udf(cols)(*cols))
	print('\nRatings by user:')
	ratings.join(movies1, on='movieId').filter('userId = {}'.format(user)).select('movieId','userId','rating','title','genres')\
		.sort('rating', ascending=False).show(20, 100)
	print('\nRecommendations for user:')
	nrecommendations.join(movies1, on='movieId').filter('userId = {}'.format(user))\
		.select('movieId','userId','rating','title','genres').show(config.nreco, 100)

# show recommendations for random a user

user = random.choice([x[0] for x in ratings.select('userId').distinct().collect()])

testing(user)
