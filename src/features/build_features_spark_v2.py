# src/features/build_features_spark_v2.py
"""
V2 do pipeline de feature engineering, reimplementado com PySpark.

A V1 (build_features.py) usa pandas -- adequada para o dataset atual
(50k linhas), mas single-node. Esta V2 reproduz a mesma lógica de negócio
(recência/frequência/intensidade, encoding categórico, imputação de nulos,
definição do target de churn) usando a DataFrame API do Spark, para que o
mesmo pipeline escale para processamento distribuído em um cluster
(particionamento, joins e agregações distribuídas) sem mudar as regras de
transformação.

Uso:
    from src.features.build_features_spark_v2 import build_features_spark
    df = build_features_spark(
        input_csv="data/raw/ecommerce_customer_churn_dataset.csv",
        output_dir="data/processed/features_spark",
    )
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


def _get_or_create_spark(app_name: str = "churn-feature-engineering") -> SparkSession:
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


CATEGORICAL_COLS = ["Gender", "Country", "City", "Payment_Method_Diversity", "Signup_Quarter"]


def _encode_categoricals(df: DataFrame, categorical_cols=CATEGORICAL_COLS) -> DataFrame:
    """Equivalente distribuído do `pd.factorize`: para cada coluna
    categórica, atribui um índice inteiro por valor distinto usando uma
    window function (dense_rank), evitando trazer os dados para o driver."""
    for col in categorical_cols:
        if col in df.columns:
            window = Window.orderBy(F.col(col).cast("string"))
            df = df.withColumn(
                col,
                F.dense_rank().over(window) - 1,
            )
    return df


def _fill_nulls(df: DataFrame, numeric_cols, categorical_cols) -> DataFrame:
    if numeric_cols:
        medians = df.approxQuantile(numeric_cols, [0.5], 0.01)
        median_map = {col: (val[0] if val else 0.0) for col, val in zip(numeric_cols, medians)}
        df = df.fillna(median_map)
    if categorical_cols:
        df = df.fillna(0, subset=[c for c in categorical_cols if c in df.columns])
    return df


def build_features_spark(
    input_csv: str = "data/raw/ecommerce_customer_churn_dataset.csv",
    output_dir: str = "data/processed/features_spark",
    churn_days: int = 30,
    spark: SparkSession = None,
) -> DataFrame:
    """
    Constrói as mesmas features da V1 (recência, frequência, duração média
    de sessão, intensidade de uso, target de churn) usando PySpark, com
    processamento distribuído (particionamento automático, agregações e
    joins via DataFrame API) no lugar do pandas single-node.
    """
    owns_spark = spark is None
    spark = spark or _get_or_create_spark()

    data = spark.read.csv(input_csv, header=True, inferSchema=True)

    # -----------------------------
    # Features numéricas básicas (equivalentes à V1)
    # -----------------------------
    numeric_defaults = {
        "recency_days": "Days_Since_Last_Purchase",
        "frequency": "Login_Frequency",
        "avg_session_duration": "Session_Duration_Avg",
        "intensity": "Pages_Per_Session",
    }
    for new_col, source_col in numeric_defaults.items():
        if source_col in data.columns:
            data = data.withColumn(new_col, F.col(source_col))
        else:
            data = data.withColumn(new_col, F.lit(0))

    data = data.withColumn("engagement_trend", F.lit(0))

    if "Churned" in data.columns:
        data = data.withColumn("churn", F.col("Churned"))
    else:
        data = data.withColumn("churn", F.lit(0))

    # -----------------------------
    # Encoding distribuído de categóricas
    # -----------------------------
    data = _encode_categoricals(data)

    # -----------------------------
    # Imputação de nulos (mediana aproximada para numéricas, particionada)
    # -----------------------------
    numeric_cols = list(numeric_defaults.keys()) + ["engagement_trend", "churn"]
    numeric_cols = [c for c in numeric_cols if c in data.columns]
    data = _fill_nulls(data, numeric_cols, CATEGORICAL_COLS)

    # -----------------------------
    # Escreve particionado (paralelo, um arquivo por partição)
    # -----------------------------
    data.write.mode("overwrite").parquet(output_dir)
    print(f"Features (Spark) salvas em {output_dir} (formato parquet particionado)")

    if owns_spark:
        # deixa a sessão aberta para o chamador inspecionar `data`; quem
        # criou a sessão é responsável por chamar spark.stop() quando
        # terminar de usar o DataFrame retornado.
        pass

    return data


def feature_summary(df: DataFrame) -> DataFrame:
    """Agregação distribuída de exemplo: estatísticas de churn por país,
    demonstrando groupBy + agregações no lugar de um `.groupby()` pandas."""
    return (
        df.groupBy("Country")
        .agg(
            F.count("*").alias("total_users"),
            F.avg("churn").alias("churn_rate"),
            F.avg("frequency").alias("avg_frequency"),
        )
        .orderBy(F.desc("total_users"))
    )


if __name__ == "__main__":
    spark = _get_or_create_spark()
    features_df = build_features_spark(spark=spark)
    feature_summary(features_df).show(20, truncate=False)
    spark.stop()
