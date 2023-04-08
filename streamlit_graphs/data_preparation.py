# data preparation for graphs
from pyspark.sql import SparkSession
import sys, glob

def main(input):
    # get the tag name
    tag_name = f'{input}'
    # integrate the data from all csv files
    folder_path = f"/Users/xxxibdara/Downloads/733-final-project/scraping/{tag_name}"

    # read all csv files in the folder
    all_files = glob.glob(folder_path + "/*.csv")

    # create an empty dataframe to store the data
    df = spark.createDataFrame([], schema="tweet_id long, user string, text string, num_comments int, num_retweets int, num_views string, timestamp timestamp")

    # read all files and append them to the dataframe
    for file_path in all_files:
        file_df = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load(file_path)
        df = df.union(file_df)

    # write the dataframe to a CSV file
    df.write.format("csv").mode("overwrite").option("header", "true").option("timestampFormat", "yyyy-MM-dd'T'HH:mm:ss.SSS'Z'").save(f"{folder_path}_all.csv")

if __name__ == '__main__':
    spark = SparkSession.builder.appName('data preparation').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    inputs = sys.argv[1]
    main(inputs)