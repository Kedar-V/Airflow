from datetime import datetime, timedelta
import os
import pandas as pd
import matplotlib.pyplot as plt
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.providers.postgres.hooks.postgres import PostgresHook
import redis  

# === Paths inside container ===
DATA_DIR = "/opt/airflow/data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = "/opt/airflow/analysis"
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)


# === Step 1: Ingest and clean customers ===
def process_customers():
    df = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    df["account_age_days"] = (pd.Timestamp("2024-08-01") - df["signup_date"]).dt.days
    df.to_csv(os.path.join(PROCESSED_DIR, "customers_clean.csv"), index=False)


# === Step 2: Ingest and clean orders ===
def process_orders():
    df = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    df["order_date"] = pd.to_datetime(df["order_date"])
    df = df[df["order_amount"] > 0]
    df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
    df.to_csv(os.path.join(PROCESSED_DIR, "orders_clean.csv"), index=False)


# === Step 3: Merge datasets ===
def merge_datasets():
    customers = pd.read_csv(os.path.join(PROCESSED_DIR, "customers_clean.csv"))
    orders = pd.read_csv(os.path.join(PROCESSED_DIR, "orders_clean.csv"))
    merged = orders.merge(customers, on="customer_id", how="inner")

    summary = (
        merged.groupby(["customer_id", "country"], as_index=False)
        .agg(
            total_spent=("order_amount", "sum"),
            num_orders=("order_id", "count"),
            avg_order_value=("order_amount", "mean"),
        )
    )
    bins = [0, 100, 250, float("inf")]
    labels = ["low", "medium", "high"]
    summary["spend_segment"] = pd.cut(summary["total_spent"], bins=bins, labels=labels)

    summary.to_csv(os.path.join(PROCESSED_DIR, "customer_order_stats.csv"), index=False)


# === Step 4: Load to Postgres ===
def load_to_postgres():
    hook = PostgresHook(postgres_conn_id="local_postgres")
    engine = hook.get_sqlalchemy_engine()
    df = pd.read_csv(os.path.join(PROCESSED_DIR, "customer_order_stats.csv"))
    df.to_sql("fact_customer_orders", engine, if_exists="replace", index=False)


# === Step 5: Analyze and visualize ===
def analyze_data():
    hook = PostgresHook(postgres_conn_id="local_postgres")
    sql = """
    SELECT country, spend_segment, AVG(avg_order_value) AS avg_aov
    FROM fact_customer_orders
    GROUP BY country, spend_segment
    ORDER BY country, spend_segment;
    """
    df = hook.get_pandas_df(sql)
    if df.empty:
        print("No data found in table")
        return
    pivot = df.pivot(index="country", columns="spend_segment", values="avg_aov")
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_title("Average Order Value by Country and Spend Segment")
    ax.set_ylabel("Average Order Value ($)")
    ax.set_xlabel("Country")
    plt.tight_layout()
    out_path = os.path.join(ANALYSIS_DIR, "avg_order_value_by_segment.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot to {out_path}")


# === Step 6: Cleanup ===
def cleanup_files():
    for f in ["customers_clean.csv", "orders_clean.csv", "customer_order_stats.csv"]:
        path = os.path.join(PROCESSED_DIR, f)
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")

def push_to_redis():
    """Push aggregated metrics from Postgres to Redis."""
    # Connect to Postgres
    hook = PostgresHook(postgres_conn_id="local_postgres")
    sql = """
    SELECT spend_segment, COUNT(*) AS num_customers, AVG(total_spent) AS avg_spend
    FROM fact_customer_orders
    GROUP BY spend_segment;
    """
    df = hook.get_pandas_df(sql)

    # Connect to Redis (service name 'redis' in docker)
    r = redis.Redis(host="redis", port=6379, db=0)

    # Push summary metrics to Redis
    for _, row in df.iterrows():
        key = f"segment:{row['spend_segment']}"
        value = {"num_customers": int(row["num_customers"]), "avg_spend": float(row["avg_spend"])}
        r.hset(key, mapping=value)
        print(f"Pushed {key} -> {value}")

    # Optionally store timestamp of last push
    r.set("last_update", datetime.utcnow().isoformat())


# === DAG definition ===
default_args = {
    "owner": "kedar",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="retail_pipeline_dag",
    default_args=default_args,
    description="Ingests, transforms, merges, loads, and analyzes retail data",
    schedule_interval="@daily",
    start_date=datetime(2024, 11, 1),
    catchup=False,
    tags=["assignment", "retail"],
) as dag:

    with TaskGroup("customers_processing") as customers_group:
        clean_customers = PythonOperator(
            task_id="process_customers", python_callable=process_customers
        )

    with TaskGroup("orders_processing") as orders_group:
        clean_orders = PythonOperator(
            task_id="process_orders", python_callable=process_orders
        )

    merge = PythonOperator(task_id="merge_datasets", python_callable=merge_datasets)
    load = PythonOperator(task_id="load_to_postgres", python_callable=load_to_postgres)
    analyze = PythonOperator(task_id="analyze_data", python_callable=analyze_data)
    cleanup = PythonOperator(task_id="cleanup_files", python_callable=cleanup_files)
    push_redis = PythonOperator(task_id="push_to_redis",python_callable=push_to_redis)


    [customers_group, orders_group] >> merge >> load >> analyze >> push_redis >> cleanup
