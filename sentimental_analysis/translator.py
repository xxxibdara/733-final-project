from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from googletrans import Translator
import col

import sys

def main():
    # Read the CSV file into a PySpark DataFrame
    df = spark.read.csv('twitter_news.csv', header=True)
    df = df.withColumnRenamed(columns={'Official   ManCity team news for  MCIBOU': 'Official ManCity team news for MCIBOU'}, inplace=True)
    text_col = col('Official ManCity team news for MCIBOU')
    
    translate_udf = udf(translate_text, StringType())
    
    # Apply the translation UDF to the DataFrame
    df = df.withColumn('translated_text', translate_udf(text_col))

    # Write the translated data to a new file
    df.write.format('csv').option('header', 'true').mode('overwrite').save('output_trans.csv')



# Define a translation function as a UDF
def translate_text(text):
    translator = Translator()
    translator.raise_Exception = True
    try:
        trans_obj = translator.translate(text, 'en')
    except Exception as err:
        translator.raise_Exception = True
        trans_obj = translator.translate(text, 'en')
    return trans_obj.text


if __name__ == '__main__':
    # Create a SparkSession
    spark = SparkSession.builder.appName('TranslationApp').getOrCreate()    
    assert sys.version_info >= (3, 5) # make sure we have Python 3.5+  
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main()
