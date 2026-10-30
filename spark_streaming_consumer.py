from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, window
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType, TimestampType

# Sesion de Spark con nivel de log WARN para reducir mensajes INFO
spark = SparkSession.builder.appName('KafkaSparkStreaming').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# Esquema de los datos de entrada
schema = StructType([
    StructField('sensor_id', IntegerType()),
    StructField('temperature', FloatType()),
    StructField('humidity', FloatType()),
    StructField('timestamp', TimestampType())
])

# Lector de streaming desde Kafka
df = (spark.readStream.format('kafka')
      .option('kafka.bootstrap.servers', 'localhost:9092')
      .option('subscribe', 'sensor_data')
      .load())

# Parsear los datos JSON
parsed_df = df.select(from_json(col('value').cast('string'), schema).alias('data')).select('data.*')

# Estadisticas por ventana de tiempo de 1 minuto y por sensor
windowed_stats = (parsed_df
                  .groupBy(window(col('timestamp'), '1 minute'), 'sensor_id')
                  .agg({'temperature': 'avg', 'humidity': 'avg'}))

# Escribir los resultados en la consola
query = (windowed_stats.writeStream
         .outputMode('complete')
         .format('console')
         .start())

query.awaitTermination()
