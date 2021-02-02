from pyspark.sql.types import StructType, StructField, IntegerType, StringType

nreco = 12

rating_schema = StructType([
	StructField("userId", IntegerType(), True),
	StructField("movieId", IntegerType(), True),
	StructField("rating", IntegerType(), True),
	StructField("timestamp", IntegerType(), True)]
)

movie_schema = StructType([
	StructField("movieId", IntegerType(), True),
	StructField("title", StringType(), True),
	StructField("release_date", StringType(), True),
	StructField("video release date", StringType(), True),
	StructField("IMDb URL", StringType(), True),
	StructField("unknown", IntegerType(), True),
	StructField("Action", IntegerType(), True),
	StructField("Adventure", IntegerType(), True),
	StructField("Animation", IntegerType(), True),
	StructField("Children's", IntegerType(), True),
	StructField("Comedy", IntegerType(), True),
	StructField("Crime", IntegerType(), True),
	StructField("Documentary", IntegerType(), True),
	StructField("Drama", IntegerType(), True),
	StructField("Fantasy", IntegerType(), True),
	StructField("Film-Noir", IntegerType(), True),
	StructField("Horror", IntegerType(), True),
	StructField("Musical", IntegerType(), True),
	StructField("Mystery", IntegerType(), True),
	StructField("Romance", IntegerType(), True),
	StructField("Sci-Fi", IntegerType(), True),
	StructField("Thriller", IntegerType(), True),
	StructField("War", IntegerType(), True),
	StructField("Western", IntegerType(), True)]
)
