from pyspark.sql import SparkSession
import pyspark.sql.functions as fn
import time
import config

spark = SparkSession.builder.appName("Recommendations_training").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")


ratings_filepath = 'data/input/u.data'
movies_filepath = 'data/input/u.item'

ratings = spark.read.option("delimiter", "\t").csv(ratings_filepath, schema=config.rating_schema)

movies = spark.read.option("delimiter", "|").csv(movies_filepath, schema=config.movie_schema)


def get_mat_sparsity(ratings):
	# Count the total number of ratings in the dataset
	count_nonzero = ratings.select("rating").count()
	# Count the number of distinct userIds and distinct movieIds
	total_elements = ratings.select("userId").distinct().count() * ratings.select("movieId").distinct().count()
	# Divide the two to get sparsity
	sparsity = (1.0 - count_nonzero/total_elements)*100
	print("The ratings dataframe is ", "%.2f" % sparsity + "% sparse.")

get_mat_sparsity(ratings)


# Create test and train set
(train, test) = ratings.randomSplit([0.8, 0.2], seed = 10)

print('\nNum records in training dataset: {}'.format(train.count()))

train.write.mode('overwrite').parquet('data/tmp/train')
test.write.mode('overwrite').parquet('data/tmp/test')


from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

# Create ALS model
als = ALS(
	userCol="userId",
	itemCol="movieId",
	ratingCol="rating",
	nonnegative = True,
	implicitPrefs = False,
	coldStartStrategy="drop"
)


# Add hyperparameters and their respective values to param_grid
param_grid = ParamGridBuilder() \
            .addGrid(als.rank, [10, 25, 50]) \
			.addGrid(als.maxIter, [5, 15]) \
            .addGrid(als.regParam, [.01, .05, 0.1]) \
            .build()


# Define evaluator as RMSE
evaluator = RegressionEvaluator(
           metricName="rmse",
           labelCol="rating",
           predictionCol="prediction")

print ("\nNumber of models to be tested: ", len(param_grid))


# Build cross validation using CrossValidator
cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

print('='*40)
print('Training ALS model')
t = time.time()
#Fit cross validator to the 'train' dataset
model = cv.fit(train)
print('\nModel fit in {} seconds.'.format(time.time()-t))

print('='*40)
print('Saving model')
model.save('als_model')

# Extract best model from the cv model above
best_model = model.bestModel


print("\n**Best Model**")
# Print "Rank"
print("  Rank:", best_model._java_obj.parent().getRank())
# Print "MaxIter"
print("  MaxIter:", best_model._java_obj.parent().getMaxIter())
# Print "RegParam"
print("  RegParam:", best_model._java_obj.parent().getRegParam())
print("\n")
