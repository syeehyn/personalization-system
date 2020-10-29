from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
import psutil
NUM_WORKER = psutil.cpu_count(logical = False)
NUM_THREAD = psutil.cpu_count(logical = True)
def Spark():
    """[summary]

    Returns:
        [type]: [description]
    """    
    conf_spark = SparkConf().set("spark.driver.host", "127.0.0.1")\
                            .set("spark.executor.instances", NUM_WORKER)\
                            .set("spark.executor.cores", int(NUM_THREAD / NUM_WORKER))\
                            .set("spark.executor.memory", '4g')\
                            .set("spark.sql.shuffle.partitions", NUM_THREAD)
    sc = SparkContext(conf = conf_spark)
    sc.setLogLevel('ERROR')
    spark = SparkSession(sc)
    print('Spark UI address {}'.format(spark.sparkContext.uiWebUrl))
    return spark