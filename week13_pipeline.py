# %% [markdown]
# # Week 13 Mini Assignment: PySpark Data Processing
#
# Question: in Q1 2023, how do NYC high-volume for-hire trip volume, fare per
# mile, and tip rate vary by borough and hour, and do airport trips look different?
#
# The source files are the TLC High Volume For-Hire Vehicle trip records for
# January–March 2023, joined to the official taxi zone lookup. Raw files are
# downloaded with `scripts/download_data.sh` and are not stored in git.

# %%
import json
import os
import time
from pathlib import Path

from pyspark import StorageLevel
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# %%
def ensure_java():
    """Point Spark at Homebrew Java 17 when the shell has no JAVA_HOME."""
    if os.environ.get("JAVA_HOME"):
        return
    candidate = Path("/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")
    if candidate.exists():
        os.environ["JAVA_HOME"] = str(candidate)
        os.environ["PATH"] = str(candidate / "bin") + os.pathsep + os.environ.get("PATH", "")


ensure_java()
# Keep the Spark UI on localhost so the query pages can be captured.
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1"

ROOT = Path.cwd()
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "output"
DOCS = ROOT / "docs"
for path in (OUT, DOCS, ROOT / "tmp" / "spark", ROOT / "spark-warehouse"):
    path.mkdir(parents=True, exist_ok=True)

# Newark, JFK, and LaGuardia in the TLC zone lookup.
AIRPORT_LOCATION_IDS = [1, 132, 138]

# %% [markdown]
# ## 1. Start Spark
#
# Local mode is enough to show a real physical plan, filter pushdown, and a
# broadcast join. Shuffle partitions are set to 16 so a laptop does not schedule
# the default 200 tiny shuffle tasks. The zone table is small, and the broadcast
# threshold is left high enough that Spark can broadcast it instead of shuffling
# every trip.

# %%
spark = (
    SparkSession.builder.appName("week13-hvfhv-q1-2023")
    .master("local[*]")
    .config("spark.driver.memory", "12g")
    .config("spark.driver.host", "127.0.0.1")
    .config("spark.driver.bindAddress", "127.0.0.1")
    .config("spark.ui.port", "4040")
    .config("spark.sql.shuffle.partitions", "16")
    .config("spark.sql.adaptive.enabled", "true")
    .config("spark.sql.autoBroadcastJoinThreshold", str(64 * 1024 * 1024))
    .config("spark.sql.parquet.filterPushdown", "true")
    .config("spark.sql.warehouse.dir", str(ROOT / "spark-warehouse"))
    .config("spark.local.dir", str(ROOT / "tmp" / "spark"))
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")
print("Spark UI:", spark.sparkContext.uiWebUrl)

# %% [markdown]
# ## 2. Load
#
# Parquet keeps column statistics inside each file. Filters written against those
# stored columns can be pushed into the scan. The zone file is a few hundred rows
# and exists only to attach borough and zone names.

# %%
trip_files = sorted(str(path) for path in RAW.glob("fhvhv_tripdata_2023-0[123].parquet"))
if len(trip_files) != 3:
    raise SystemExit(f"Expected 3 monthly Parquet files in {RAW}, found {trip_files}")

raw_bytes = sum(Path(path).stat().st_size for path in trip_files)
print(f"Input Parquet size: {raw_bytes / 1024**3:.2f} GiB across {len(trip_files)} files")
if raw_bytes < 1024**3:
    raise SystemExit("Input is under 1 GiB. Download all three months before running.")

trips = spark.read.parquet(*trip_files)

required_trip_columns = {
    "pickup_datetime",
    "PULocationID",
    "DOLocationID",
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tips",
    "driver_pay",
    "shared_match_flag",
}
missing = required_trip_columns - set(trips.columns)
if missing:
    raise SystemExit(f"Trip files are missing columns: {sorted(missing)}")

zones = (
    spark.read.option("header", True)
    .option("inferSchema", True)
    .csv(str(RAW / "taxi_zone_lookup.csv"))
    .withColumn("LocationID", F.col("LocationID").cast("int"))
)
print("Trip columns:", trips.columns)
trips.printSchema()
zones.printSchema()

# %% [markdown]
# ## 3. Transformations are lazy
#
# The cell below only records a plan: two filters, a narrow projection,
# derived columns, two broadcast joins, and a groupBy. None of that reads the
# trip files. The clock should stay near zero, and no job is launched until a
# later action.

# %%
t0 = time.perf_counter()

# Filters stay on stored columns, before casts and derived fields, so the
# Parquet scan can skip row groups. Two filter steps: data-quality checks,
# then the Q1 window and physical bounds.
valid_trips = trips.filter(
    (F.col("trip_miles") > 0)
    & (F.col("trip_time") > 0)
    & (F.col("base_passenger_fare") > 0)
    & (F.col("tips") >= 0)
    & (F.col("driver_pay") >= 0)
    & F.col("PULocationID").isNotNull()
    & F.col("DOLocationID").isNotNull()
    & F.col("pickup_datetime").isNotNull()
)
bounded_trips = valid_trips.filter(
    (F.col("trip_miles") < 200)
    & (F.col("trip_time") < 6 * 60 * 60)
    & (F.col("base_passenger_fare") < 1000)
    & (F.col("pickup_datetime") >= F.to_timestamp(F.lit("2023-01-01")))
    & (F.col("pickup_datetime") < F.to_timestamp(F.lit("2023-04-01")))
)

# Column pruning happens before the join. trip_time is seconds in the TLC
# high-volume dictionary, so minutes and fare per mile are derived only after
# the bad rows are gone.
pruned = bounded_trips.select(
    "pickup_datetime",
    F.col("PULocationID").cast("int").alias("PULocationID"),
    F.col("DOLocationID").cast("int").alias("DOLocationID"),
    "trip_miles",
    "trip_time",
    "base_passenger_fare",
    "tips",
    "driver_pay",
    F.upper(F.trim(F.col("shared_match_flag"))).alias("shared_match_flag"),
)
enriched = (
    pruned.withColumn("pickup_hour", F.hour("pickup_datetime"))
    .withColumn("trip_minutes", F.col("trip_time") / F.lit(60.0))
    .withColumn("fare_per_mile", F.col("base_passenger_fare") / F.col("trip_miles"))
    .withColumn("tip_rate", F.col("tips") / F.col("base_passenger_fare"))
    .withColumn(
        "is_shared",
        F.when(F.col("shared_match_flag") == "Y", F.lit(1.0)).otherwise(F.lit(0.0)),
    )
)

pickup_zones = zones.select(
    F.col("LocationID").alias("PULocationID"),
    F.col("Borough").alias("pickup_borough"),
    F.col("Zone").alias("pickup_zone"),
)
dropoff_zones = zones.select(
    F.col("LocationID").alias("DOLocationID"),
    F.col("Borough").alias("dropoff_borough"),
)

# Broadcast the few hundred zone rows. Repartitioning the trip table here
# would shuffle tens of millions of rows for no gain.
joined = (
    enriched.join(F.broadcast(pickup_zones), "PULocationID", "left")
    .join(F.broadcast(dropoff_zones), "DOLocationID", "left")
    .withColumn("pickup_borough", F.coalesce(F.col("pickup_borough"), F.lit("Unknown")))
    .withColumn("dropoff_borough", F.coalesce(F.col("dropoff_borough"), F.lit("Unknown")))
    .withColumn(
        "is_airport",
        F.col("PULocationID").isin(AIRPORT_LOCATION_IDS)
        | F.col("DOLocationID").isin(AIRPORT_LOCATION_IDS),
    )
    .withColumn(
        "same_borough",
        F.when(F.col("pickup_borough") == F.col("dropoff_borough"), F.lit(1.0)).otherwise(F.lit(0.0)),
    )
    # Cache and later queries only need the analysis columns. Dropping the
    # raw timestamps, ids, and flags here keeps the cached table smaller.
    .select(
        "pickup_borough",
        "pickup_zone",
        "pickup_hour",
        "is_airport",
        "same_borough",
        "trip_miles",
        "trip_minutes",
        "base_passenger_fare",
        "fare_per_mile",
        "tip_rate",
        "is_shared",
        "driver_pay",
    )
)
joined.createOrReplaceTempView("trips")

borough_hour = (
    joined.groupBy("pickup_borough", "pickup_hour")
    .agg(
        F.count("*").alias("trip_count"),
        F.avg("trip_miles").alias("avg_miles"),
        F.avg("fare_per_mile").alias("avg_fare_per_mile"),
        F.avg("tip_rate").alias("avg_tip_rate"),
        F.avg("is_shared").alias("shared_trip_rate"),
        F.sum("driver_pay").alias("total_driver_pay"),
    )
    .orderBy("pickup_borough", "pickup_hour")
)

# Two SQL queries over the same cleaned view. SQL is planned by the same
# optimizer as the DataFrame above.
sql_borough = spark.sql(
    """
    SELECT
      pickup_borough,
      COUNT(*) AS trips,
      ROUND(SUM(base_passenger_fare), 2) AS total_base_fare,
      ROUND(AVG(tip_rate), 4) AS avg_tip_rate,
      ROUND(AVG(fare_per_mile), 2) AS avg_fare_per_mile,
      ROUND(AVG(trip_miles), 2) AS avg_miles,
      ROUND(AVG(is_shared), 4) AS shared_trip_rate,
      ROUND(AVG(same_borough), 4) AS same_borough_rate
    FROM trips
    GROUP BY pickup_borough
    ORDER BY trips DESC
    """
)
sql_airport = spark.sql(
    """
    SELECT
      pickup_hour,
      is_airport,
      COUNT(*) AS trips,
      ROUND(AVG(fare_per_mile), 2) AS avg_fare_per_mile,
      ROUND(AVG(tip_rate), 4) AS avg_tip_rate,
      ROUND(AVG(trip_miles), 2) AS avg_miles,
      ROUND(AVG(trip_minutes), 2) AS avg_minutes
    FROM trips
    GROUP BY pickup_hour, is_airport
    ORDER BY pickup_hour, is_airport
    """
)

transform_seconds = time.perf_counter() - t0
print(
    f"Transformations built in {transform_seconds:.4f}s. "
    "No action has run, so Spark has not scanned the trip files."
)

# %% [markdown]
# ## 4. Physical plan
#
# `explain` asks the optimizer for the plan it will run. It does not compute the
# aggregations. The formatted plan should show pushed Parquet filters and
# `BroadcastHashJoin` for both zone lookups.

# %%
def explain_formatted(df):
    jvm = df.sparkSession._jvm
    try:
        return jvm.PythonSQLUtils.explainString(df._jdf.queryExecution(), "formatted")
    except Exception:
        mode = jvm.org.apache.spark.sql.execution.ExplainMode.fromString("formatted")
        return df._jdf.queryExecution().explainString(mode)


plan_sections = [
    ("Borough-hour aggregation", borough_hour),
    ("SQL: airport trips by hour", sql_airport),
]
plan_text = []
for title, frame in plan_sections:
    text = explain_formatted(frame)
    print(f"\n===== {title} =====\n")
    print(text)
    plan_text.append(f"===== {title} =====\n{text}\n")

plan_path = DOCS / "explain_plan.txt"
plan_path.write_text("\n".join(plan_text))
print("Wrote", plan_path)

# %% [markdown]
# ## 5. Actions are eager
#
# `.collect()` is the action. The first call launches a job and scans the files.
# The second uncached call plans a new job and scans them again, because nothing
# was stored. That gap is the difference between a transformation and an action.

# %%
t0 = time.perf_counter()
borough_rows = [row.asDict() for row in sql_borough.collect()]
uncached_borough_s = time.perf_counter() - t0

t0 = time.perf_counter()
airport_rows = [row.asDict() for row in sql_airport.collect()]
uncached_airport_s = time.perf_counter() - t0

print(f"Action 1, uncached borough SQL: {uncached_borough_s:.2f}s")
print(f"Action 2, uncached airport SQL: {uncached_airport_s:.2f}s")

# %% [markdown]
# ## 6. Cache the cleaned trips and repeat the actions
#
# `.cache()` is itself lazy. The next action pays to build the cache. The action
# after that should reuse the cached trips instead of scanning Parquet again.
# The cleaned trip table is the frame worth caching: both SQL queries read it.
# The tiny aggregates are not.

# %%
joined.persist(StorageLevel.MEMORY_AND_DISK)

t0 = time.perf_counter()
sql_borough.collect()
cache_build_s = time.perf_counter() - t0

t0 = time.perf_counter()
sql_airport.collect()
cached_airport_s = time.perf_counter() - t0

cleaned_rows = joined.count()
print(f"Action 3, first action after cache(): {cache_build_s:.2f}s (computes and stores)")
print(f"Action 4, second action after cache(): {cached_airport_s:.2f}s (reuses the cache)")
print(f"Cleaned trip rows: {cleaned_rows:,}")
print("Borough summary:")
sql_borough.show(truncate=False)
print("Airport comparison, first hours:")
sql_airport.show(8, truncate=False)
print(
    "Repeated-action speedup: "
    f"{uncached_airport_s / cached_airport_s:.2f}x "
    f"({uncached_airport_s:.2f}s uncached vs {cached_airport_s:.2f}s cached)"
)

top_zones = (
    joined.groupBy("pickup_borough", "pickup_zone")
    .agg(
        F.count("*").alias("trips"),
        F.round(F.avg("fare_per_mile"), 2).alias("avg_fare_per_mile"),
    )
    .orderBy(F.desc("trips"))
    .limit(10)
)
top_zone_rows = [row.asDict() for row in top_zones.collect()]
print("Busiest pickup zones:")
top_zones.show(truncate=False)

hour_rows = [row.asDict() for row in borough_hour.collect()]

# %% [markdown]
# ## 7. Write Parquet
#
# The result is a few hundred rows, so the partition column is the borough on
# that summary, not the raw trip table. Partitioning the raw trips would be a
# full shuffle of every ride just to name the output folders.

# %%
borough_path = OUT / "borough_hour_summary"
airport_path = OUT / "airport_by_hour"
borough_hour.write.mode("overwrite").partitionBy("pickup_borough").parquet(str(borough_path))
sql_airport.write.mode("overwrite").parquet(str(airport_path))
print("Wrote", borough_path)
print("Wrote", airport_path)

# %% [markdown]
# ## 8. Numbers used in the write-up
#
# These are the values the README quotes. They come from the actions above.

# %%
def weighted(rows, airport_flag):
    subset = [row for row in rows if bool(row["is_airport"]) is airport_flag]
    trips_n = sum(row["trips"] for row in subset)
    if trips_n == 0:
        return {"trips": 0, "avg_fare_per_mile": None, "avg_tip_rate": None, "avg_miles": None}
    return {
        "trips": trips_n,
        "avg_fare_per_mile": sum(row["avg_fare_per_mile"] * row["trips"] for row in subset) / trips_n,
        "avg_tip_rate": sum(row["avg_tip_rate"] * row["trips"] for row in subset) / trips_n,
        "avg_miles": sum(row["avg_miles"] * row["trips"] for row in subset) / trips_n,
    }


hour_totals = {}
for row in hour_rows:
    hour = int(row["pickup_hour"])
    hour_totals[hour] = hour_totals.get(hour, 0) + int(row["trip_count"])
peak_hour = max(hour_totals, key=hour_totals.get)
busiest = max(borough_rows, key=lambda row: row["trips"])

metrics = {
    "input_bytes": raw_bytes,
    "input_gib": round(raw_bytes / 1024**3, 3),
    "cleaned_rows": cleaned_rows,
    "transform_seconds": round(transform_seconds, 4),
    "uncached_borough_seconds": round(uncached_borough_s, 2),
    "uncached_airport_seconds": round(uncached_airport_s, 2),
    "cache_build_seconds": round(cache_build_s, 2),
    "cached_airport_seconds": round(cached_airport_s, 2),
    "borough_summary": borough_rows,
    "airport_vs_not": {
        "airport": weighted(airport_rows, True),
        "not_airport": weighted(airport_rows, False),
    },
    "peak_hour": peak_hour,
    "peak_hour_trips": hour_totals[peak_hour],
    "busiest_borough": busiest["pickup_borough"],
    "top_zones": top_zone_rows,
}
metrics_path = DOCS / "metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2, default=str))
print("Busiest borough:", busiest["pickup_borough"], "trips:", busiest["trips"])
print("Peak pickup hour:", peak_hour, "trips:", hour_totals[peak_hour])
print("Airport vs not:", json.dumps(metrics["airport_vs_not"], default=str))
print("Wrote", metrics_path)

# %%
joined.unpersist()
if os.environ.get("SCREENSHOT_PAUSE") == "1":
    ready = DOCS / "screenshots_done"
    ui_file = DOCS / "spark_ui_ready"
    if ready.exists():
        ready.unlink()
    ui_file.write_text((spark.sparkContext.uiWebUrl or "http://127.0.0.1:4040") + "\n")
    print("SPARK_UI_READY", spark.sparkContext.uiWebUrl, flush=True)
    deadline = time.time() + 420
    while time.time() < deadline and not ready.exists():
        time.sleep(2)
    ui_file.unlink(missing_ok=True)
    print("Screenshot pause finished", flush=True)
spark.stop()
