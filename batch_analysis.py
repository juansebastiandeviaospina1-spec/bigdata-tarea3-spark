# batch_analysis.py - Tarea 3 Big Data (UNAD)
# Analisis batch de casos positivos de COVID-19 en Colombia (INS, datos.gov.co)
# Uso: spark-submit batch_analysis.py datos/dataset.csv
import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import matplotlib
matplotlib.use('Agg')  # sin pantalla: guarda las graficas como PNG
import matplotlib.pyplot as plt

RUTA_CSV = sys.argv[1] if len(sys.argv) > 1 else 'datos/dataset.csv'
SALIDAS = 'salidas'

spark = SparkSession.builder.appName('Tarea3_Batch_COVID').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# 1. CARGA desde la fuente original (todo como texto; los tipos se castean abajo)
df = spark.read.csv(RUTA_CSV, header=True, inferSchema=False)
print('Filas cargadas:', df.count(), '| Columnas:', len(df.columns))
df.printSchema()
df.select('fecha_reporte_web', 'departamento_nom', 'ciudad_municipio_nom',
          'edad', 'unidad_medida', 'sexo', 'estado').show(5, truncate=False)

# 2. LIMPIEZA
df = df.dropDuplicates(['id_de_caso'])
cols_clave = ['departamento_nom', 'edad', 'fecha_reporte_web', 'sexo']
print('Nulos en columnas clave:')
df.select([F.sum(F.col(c).isNull().cast('int')).alias(c) for c in cols_clave]).show()
df = df.dropna(subset=cols_clave)

# 3. TRANSFORMACION
df = (df.withColumn('edad', F.col('edad').cast('int'))
        .withColumn('unidad_medida', F.col('unidad_medida').cast('int'))
        .withColumn('fecha_reporte', F.to_date(F.col('fecha_reporte_web')))
        .withColumn('anio', F.year('fecha_reporte'))
        .withColumn('mes', F.month('fecha_reporte'))
        .withColumn('departamento_nom', F.upper(F.trim(F.col('departamento_nom')))))
df = df.withColumn('sexo', F.upper(F.trim(F.col('sexo')))).withColumn('estado', F.initcap(F.trim(F.col('estado'))))
# unidad_medida: 1 = anios, 2 = meses, 3 = dias (menores de 1 anio cuentan como 0)
df = df.withColumn('edad_anios',
                   F.when(F.col('unidad_medida') == 1, F.col('edad')).otherwise(F.lit(0)))
df = df.filter((F.col('edad_anios') >= 0) & (F.col('edad_anios') <= 110))
df = df.withColumn('grupo_edad',
                   F.when(F.col('edad_anios') < 18, '0-17')
                    .when(F.col('edad_anios') < 30, '18-29')
                    .when(F.col('edad_anios') < 45, '30-44')
                    .when(F.col('edad_anios') < 60, '45-59')
                    .otherwise('60+'))
df.cache()
print('Filas despues de limpiar:', df.count())

# 4. EDA con DataFrames
print('Resumen de la edad (anios):')
df.describe('edad_anios').show()

por_depto = (df.groupBy('departamento_nom')
               .agg(F.count('*').alias('casos'),
                    F.round(F.avg('edad_anios'), 1).alias('edad_promedio'))
               .orderBy(F.desc('casos')))
print('Top 10 departamentos por casos:')
por_depto.show(10, truncate=False)

por_mes = (df.groupBy('anio', 'mes').agg(F.count('*').alias('casos'))
             .orderBy('anio', 'mes'))
print('Casos por mes:')
por_mes.show(36)

print('Casos por sexo:')
df.groupBy('sexo').count().orderBy(F.desc('count')).show()
print('Casos por estado:')
df.groupBy('estado').count().orderBy(F.desc('count')).show()
por_edad = df.groupBy('grupo_edad').count().orderBy('grupo_edad')
print('Casos por grupo de edad:')
por_edad.show()

# 5. EDA con RDDs (transformaciones y acciones de la guia)
rdd = df.select('departamento_nom').rdd.map(lambda r: (r[0], 1))   # map
conteo = rdd.reduceByKey(lambda a, b: a + b)                        # reduceByKey
grandes = conteo.filter(lambda x: x[1] > 10000)                     # filter
print('Departamentos distintos (count):', conteo.count())           # count
print('Departamentos con mas de 10.000 casos (collect):', grandes.collect())  # collect
print('Top 5 (take):', conteo.sortBy(lambda x: -x[1]).take(5))     # take
print('Total de casos (reduce):',
      conteo.map(lambda x: x[1]).reduce(lambda a, b: a + b))        # reduce

# 6. VISUALIZACION
top = por_depto.limit(10).toPandas()
plt.figure(figsize=(10, 5))
plt.barh(top['departamento_nom'].astype(str), top['casos'])
plt.gca().invert_yaxis()
plt.title('Top 10 departamentos por casos de COVID-19')
plt.xlabel('Casos')
plt.tight_layout()
plt.savefig(SALIDAS + '/top_departamentos.png')
plt.close()

mes = por_mes.toPandas()
etiquetas = mes['anio'].astype(str) + '-' + mes['mes'].astype(str).str.zfill(2)
plt.figure(figsize=(11, 5))
plt.plot(etiquetas, mes['casos'], marker='o')
plt.xticks(rotation=90)
plt.title('Casos de COVID-19 por mes de reporte')
plt.ylabel('Casos')
plt.tight_layout()
plt.savefig(SALIDAS + '/casos_por_mes.png')
plt.close()

edades = por_edad.toPandas()
plt.figure(figsize=(7, 5))
plt.bar(edades['grupo_edad'], edades['count'])
plt.title('Casos por grupo de edad')
plt.ylabel('Casos')
plt.tight_layout()
plt.savefig(SALIDAS + '/casos_por_grupo_edad.png')
plt.close()

print('Graficas guardadas en la carpeta', SALIDAS)
spark.stop()
