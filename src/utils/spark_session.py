from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import psutil
NUM_WORKER = psutil.cpu_count(logical = False)

def Spark():
    """[summary]

    Returns:
        [type]: [description]
    """    
    conf_spark = SparkConf().set("spark.driver.host", "127.0.0.1")
    sc = SparkContext(conf = conf_spark)
    spark = SparkSession(sc)
    spark.conf.set("spark.sql.shuffle.partitions", NUM_WORKER)
    print('Spark UI address {}'.format(spark.sparkContext.uiWebUrl))
    return spark