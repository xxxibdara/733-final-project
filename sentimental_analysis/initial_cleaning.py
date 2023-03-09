import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+
from pyspark.sql import SparkSession, functions, types
from pyspark.sql.types import *
from pyspark.sql.functions import  regexp_replace,col



def main(inputs, output):
    
  
    df = spark.read.option("header", "true").csv(inputs)

    df = df.select('content','hashtags','likeCount')
    
    df = df.withColumn("content", regexp_replace(col("content"), "http\S+", ""))
    df = df.withColumn("content", regexp_replace("content", "[^a-zA-Z0-9\ ]", " "))
    df = df.withColumn("hashtags", regexp_replace("hashtags", "[^a-zA-Z0-9\ ]", " "))

    

    df = df.filter(df['hashtags']!='None')
    df = df.filter(df['hashtags']!="")
    df = df.filter(df['content']!="")
    df = df.filter(df['likeCount']!=0)

    

    df.sort('likeCount', ascending=False).write.csv(output, mode='overwrite')
    

if __name__ == '__main__':
    spark = SparkSession.builder.appName('yelp').getOrCreate()
    assert spark.version >= '3.0' # make sure we have Spark 3.0+
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    inputs = sys.argv[1]
    output = sys.argv[2]
    main(inputs, output)



    
