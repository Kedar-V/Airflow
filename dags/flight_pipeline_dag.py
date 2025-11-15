from datetime import datetime, timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.providers.postgres.hooks.postgres import PostgresHook
import redis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# === Directories ===
DATA_DIR = "/opt/airflow/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = "/opt/airflow/analysis"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# === Step 1a: Ingest flights.csv (10% sample) ===
def ingest_flights(**context):
    path = os.path.join(DATA_DIR, "flights.csv")
    df = pd.read_csv(path)

    cols = [
        "year", "month", "day", "dep_time", "dep_delay",
        "arr_time", "arr_delay", "carrier", "origin", "dest"
    ]
    df = df[cols].dropna(subset=["dep_delay", "arr_delay"])
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)

    out_path = os.path.join(PROCESSED_DIR, "flights_raw.parquet")
    df.to_parquet(out_path, index=False)
    context["ti"].xcom_push(key="flights_raw", value=out_path)
    print(f"📉 Flight sample size: {len(df)}")


# === Step 1b: Ingest weather.csv ===
def ingest_weather(**context):
    path = os.path.join(DATA_DIR, "weather.csv")
    df = pd.read_csv(path)
    out_path = os.path.join(PROCESSED_DIR, "weather.parquet")
    df.to_parquet(out_path, index=False)
    context["ti"].xcom_push(key="weather_path", value=out_path)
    print(f"🌤️ Weather ingested → {out_path}")


# === Step 2a: Process departure delay ===
def process_dep_delay(**context):
    in_path = context["ti"].xcom_pull(key="flights_raw")
    df = pd.read_parquet(in_path)
    df_dep = df[["year", "month", "day", "dep_delay"]].copy()
    df_dep["dep_status"] = df_dep["dep_delay"].apply(
        lambda x: "early" if x < 0 else ("on_time" if x <= 15 else "late")
    )
    out_path = os.path.join(PROCESSED_DIR, "dep_delay.parquet")
    df_dep.to_parquet(out_path, index=False)
    context["ti"].xcom_push(key="dep_path", value=out_path)


# === Step 2b: Process arrival delay ===
def process_arr_delay(**context):
    in_path = context["ti"].xcom_pull(key="flights_raw")
    df = pd.read_parquet(in_path)
    df_arr = df[["year", "month", "day", "arr_delay"]].copy()
    df_arr["arr_status"] = df_arr["arr_delay"].apply(
        lambda x: "early" if x < 0 else ("on_time" if x <= 15 else "late")
    )
    out_path = os.path.join(PROCESSED_DIR, "arr_delay.parquet")
    df_arr.to_parquet(out_path, index=False)
    context["ti"].xcom_push(key="arr_path", value=out_path)


# === Step 3: Merge dep + arr + weather ===
def merge_delays(**context):
    dep_path = context["ti"].xcom_pull(key="dep_path")
    arr_path = context["ti"].xcom_pull(key="arr_path")
    weather_path = context["ti"].xcom_pull(key="weather_path")

    dep = pd.read_parquet(dep_path)
    arr = pd.read_parquet(arr_path)
    weather = pd.read_parquet(weather_path)

    merged = (
        dep
        .merge(arr, on=["year", "month", "day"], how="inner")
        .merge(weather, on=["year", "month", "day"], how="left")
    )

    merged["total_delay"] = merged["dep_delay"] + merged["arr_delay"]
    merged["delay_category"] = merged["total_delay"].apply(
        lambda x: "early" if x < 0 else ("on_time" if x <= 15 else "late")
    )

    out_path = os.path.join(PROCESSED_DIR, "flights_merged.parquet")
    merged.to_parquet(out_path, index=False)
    context["ti"].xcom_push(key="merged_path", value=out_path)

    print(f"🔀 Merged flights + weather: {len(merged)} rows")


# === Step 4: Load to Postgres ===
def load_to_postgres(**context):
    hook = PostgresHook(postgres_conn_id="local_postgres")
    engine = hook.get_sqlalchemy_engine()
    merged_path = context["ti"].xcom_pull(key="merged_path")
    df = pd.read_parquet(merged_path)
    df = df.sample(frac=0.1, random_state=42).reset_index(drop=True)
    print(f"📉 Loading only 10% to Postgres → {len(df)} rows")
    
    df.to_sql("fact_flights", engine, if_exists="replace", index=False)
    print(f"💾 Loaded {len(df)} rows into Postgres")


# === Step 5: Analysis ===
def analyze_data(**context):
    hook = PostgresHook(postgres_conn_id="local_postgres")
    sql = """
        SELECT month, delay_category, COUNT(*) AS num_flights, AVG(total_delay) AS avg_delay
        FROM fact_flights
        GROUP BY month, delay_category
        ORDER BY month;
    """
    df = hook.get_pandas_df(sql)

    if df.empty:
        print("⚠️ No analysis data.")
        return

    pivot = df.pivot(index="month", columns="delay_category", values="avg_delay")
    ax = pivot.plot(kind="bar", figsize=(8, 4))
    ax.set_title("Average Delay by Month (10% Sample)")
    ax.set_ylabel("Avg Delay (min)")
    plt.tight_layout()
    path = os.path.join(ANALYSIS_DIR, "avg_delay_by_month.png")
    plt.savefig(path)
    plt.close()


# === Step 6: ML Model ===
def train_delay_model(**context):
    merged_path = context["ti"].xcom_pull(key="merged_path")
    df = pd.read_parquet(merged_path)

    df["target"] = df["delay_category"].map({"early": 0, "on_time": 1, "late": 2})
    features = ["month", "dep_delay", "arr_delay"]
    X = df[features]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=200)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    print(classification_report(y_test, preds))

    joblib.dump(model, os.path.join(ANALYSIS_DIR, "delay_model.pkl"))
    joblib.dump(scaler, os.path.join(ANALYSIS_DIR, "delay_scaler.pkl"))


# === Step 7: Push summary to Redis ===
def push_to_redis(**context):
    hook = PostgresHook(postgres_conn_id="local_postgres")
    sql = """SELECT delay_category, COUNT(*) AS count FROM fact_flights GROUP BY delay_category;"""
    df = hook.get_pandas_df(sql)

    r = redis.Redis(host="redis", port=6379, db=0)

    for _, row in df.iterrows():
        r.hset(f"delay:{row['delay_category']}", mapping={"count": int(row["count"])})

    r.set("last_update", datetime.utcnow().isoformat())


# === Step 8: Cleanup ===
def cleanup_files():
    for f in os.listdir(PROCESSED_DIR):
        if f.endswith(".parquet"):
            os.remove(os.path.join(PROCESSED_DIR, f))


# === DAG Definition ===
default_args = {"owner": "kedar", "retries": 1, "retry_delay": timedelta(minutes=5)}

with DAG(
    dag_id="flights_pipeline_dag_final_sampled",
    default_args=default_args,
    description="Parallel ingest of flights + weather, ML, Redis, Postgres",
    schedule="@daily",
    start_date=datetime(2024, 11, 1),
    catchup=False,
    tags=["flights", "ml", "weather"],
) as dag:

    ingest = PythonOperator(task_id="ingest_flights", python_callable=ingest_flights)
    ingest_weather_task = PythonOperator(task_id="ingest_weather", python_callable=ingest_weather)

    with TaskGroup("delay_processing") as delay_group:
        dep = PythonOperator(task_id="process_dep_delay", python_callable=process_dep_delay)
        arr = PythonOperator(task_id="process_arr_delay", python_callable=process_arr_delay)
        merge = PythonOperator(task_id="merge_delays", python_callable=merge_delays)
        [dep, arr] >> merge

    load = PythonOperator(task_id="load_to_postgres", python_callable=load_to_postgres)
    analyze = PythonOperator(task_id="analyze_data", python_callable=analyze_data)
    train_model = PythonOperator(task_id="train_delay_model", python_callable=train_delay_model)
    push_redis = PythonOperator(task_id="push_to_redis", python_callable=push_to_redis)
    cleanup = PythonOperator(task_id="cleanup_files", python_callable=cleanup_files)

    # **Parallel ingest → merge → load → analysis/ML → redis → cleanup**
    [ingest, ingest_weather_task] >> delay_group >> load
    load >> [analyze, train_model]
    [analyze, train_model] >> push_redis >> cleanup