from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType
from googletrans import Translator

import sys, re

def main():
    # Read the CSV file into a PySpark DataFrame
    df = spark.read.csv("sample.csv", header=True)
    text_col = col("Official ManCity team news for MCIBOU")
    
    translate_udf = udf(translate_text, StringType())
    
    # Apply the translation UDF to the DataFrame
    df = df.withColumn("translated_text", translate_udf(df['text']))

    # Write the translated data to a new file
    df.write.format("csv").option("header", "true").mode("overwrite").save("output_trans.csv")



# Define a translation function as a UDF
def translate_text(text):
    translator = Translator(service_urls=['translate.google.com']) 
    text = re.sub("[^a-zA-Z]", " ", text)
    try:
        translated_text = translator.translate(text, dest='en').text
        return translated_text
    except Exception as e:
        print(f"Translation failed: {e}")
        return ""


if __name__ == '__main__':
    # Create a SparkSession
    spark = SparkSession.builder.appName("TranslationApp").getOrCreate()    
    assert sys.version_info >= (3, 5) # make sure we have Python 3.5+  
    spark.sparkContext.setLogLevel('WARN')
    sc = spark.sparkContext
    main()
