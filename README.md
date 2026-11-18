# Week 13 Mini Assignment: PySpark Data Processing

NYC high-volume for-hire trips for the first quarter of 2023, processed with PySpark. The pipeline filters bad rows before any join, broadcasts the taxi-zone lookup, aggregates by borough and hour, runs two SQL queries, and writes Parquet summaries.

## Dataset

The trip files are the TLC High Volume For-Hire Vehicle records (Uber, Lyft, and Via trips reported to the NYC Taxi and Limousine Commission). The zone file is the official taxi zone lookup.

| File | Size |
|---|---|
| `fhvhv_tripdata_2023-01.parquet` | 465 MB |
| `fhvhv_tripdata_2023-02.parquet` | 449 MB |
| `fhvhv_tripdata_2023-03.parquet` | 513 MB |
| `taxi_zone_lookup.csv` | 265 zones |

Together the Parquet files are **1.36 GiB**. They are not in git. Download them with `scripts/download_data.sh`.

Source: [TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page). The files are the public copies on the TLC CloudFront bucket (`d37ci6vzurychx.cloudfront.net`). `trip_time` is elapsed seconds, from the TLC high-volume data dictionary. Airport zones are Newark (location 1), JFK (132), and LaGuardia (138).

## How to run

Java 17 is required. From the repo root, with PySpark 4.3 installed (`requirements.txt`):

```bash
./scripts/download_data.sh
python week13_pipeline.py
```

Or execute `week13_pipeline.ipynb` and export HTML with `jupyter nbconvert --to html week13_pipeline.ipynb`. The notebook is the same pipeline as the script. Run it from the repo root so `data/raw` resolves.

The script keeps shuffle partitions at 16, which is enough for a laptop and avoids the default 200 tiny tasks. It does not repartition the trip table. The only partition column is `pickup_borough` on the small summary that gets written.

## Same ideas as the class session

The [Databricks session](https://androidilicious.github.io/databricks-pyspark-session/) walks through lazy plans, `explain()`, columnar files, broadcast joins, and a window plus SQL that return the same rows. This notebook uses those same steps on the TLC data:

| Session step | Where it shows up here |
|---|---|
| Lazy plan, then an action | Section 3 prints the DataFrame before any job. Section 5 times the first `collect()`. |
| `explain()`, `ReadSchema`, filters pushed into the scan | Section 4. Local Spark prints `DataFilters` and `PushedFilters`. Photon on Databricks prints `DataFilters` for the same thing. |
| Broadcast a small lookup | Both zone joins use `F.broadcast`. The plan says `BroadcastHashJoin`. |
| Window, then the same answer in SQL | Section 7 ranks the busiest hour inside each borough both ways. |
| Write a columnar table | Section 8 writes Parquet, partitioned by borough. Parquet is what makes the filter pushdown possible, which is the point of the session's Delta step. |

The session runs on Databricks serverless, where `spark` already exists. This repo runs the same API locally because the write-up needs a Spark UI capture and a GitHub repo. `scripts/download_data.sh` is the stand-in for the built-in `/databricks-datasets` path.

## What the pipeline does

1. Read the three Parquet files and the zone CSV.
2. Two filters on stored columns, before any derived column: drop non-positive miles, time, or fare, and negative tips or driver pay; then keep physically plausible trips inside Q1 2023 (under 200 miles, under 6 hours, base fare under $1,000).
3. Keep the nine columns the rest of the job uses. Add hour, minutes, fare per mile, tip rate, and a shared-ride flag with `withColumn`.
4. Broadcast-join pickup zone and dropoff zone. Flag airport trips and same-borough trips.
5. `groupBy` pickup borough and hour.
6. Two SQL queries on the view `trips`: borough summary, and airport versus other trips by hour.
7. Write `data/output/borough_hour_summary` (partitioned by borough) and `data/output/airport_by_hour`.

## Performance analysis

Catalyst did the three optimizations the pipeline was written to get. The formatted plan in `docs/explain_plan.txt`, and the SQL details page for query 3, show a Parquet scan whose `PushedFilters` list includes the mile, duration, fare, tip, driver-pay, location, and Q1 timestamp predicates. Those predicates are on stored columns, so Spark can skip Parquet row groups instead of decoding every record and filtering afterward. The same scan's `ReadSchema` is nine fields, not the 24 columns stored in the files. Both zone joins are `BroadcastHashJoin` with a `BroadcastExchange`. The zone table is 265 rows, and broadcasting it means the 57 million trips are not shuffled just to attach a borough name. Adaptive query execution then does the hash aggregate.

The lineage itself is lazy. Building every filter, projection, join, and both SQL queries took **0.41 seconds** and launched no job. Printing the DataFrame at that point shows only its schema, the same check as the session's "Nothing has run yet." The first action, the borough SQL, took **9.7 seconds**. The scan in that plan reads about 56.8 million rows, then filters, broadcast-joins, and aggregates. The airport SQL is a different aggregation on the same lineage and took **6.4 seconds**, so Spark ran the plan again. It was faster than a cold read because the operating system already had the Parquet pages cached. That speedup is the file cache, not a Spark `.cache()`. The saved SQL screenshot is an earlier pass of this same plan: query 3 completed in 9 seconds with the scan, filter, and both broadcast joins on the details page.

`.persist(MEMORY_AND_DISK)` is also lazy. The action that stores the trips is `count()`, which took **39 seconds**. On the earlier UI capture that same step was query 7, and its plan was an `InMemoryRelation` holding **5.3 GiB and 56.8 million rows**. That is the bottleneck. The compressed input is 1.36 GiB, and the cached table is about four times larger, so a later pass over the cache costs about as much as reading Parquet again. Repeating the borough and airport SQL took **0.03 and 0.04 seconds** because those queries return a few rows, not because every trip was read again. On this laptop, the optimizations that removed work on every scan were the early filters, the narrow column list, and the broadcast joins. Caching every cleaned trip made the working set bigger.

## Key findings

The cleaned data has **56,799,186** trips.

Manhattan is the origin for 23.4 million of them (41 percent), then Brooklyn (14.7 million), Queens (11.1 million), the Bronx (6.8 million), and Staten Island (0.77 million). Another 2,920 trips have borough `N/A`, and only 3 pickups are coded to Newark, so that Newark fare-per-mile average is not a real pattern. The busiest hour across the city is **6 p.m.**, with 3.5 million pickups across the quarter. A window rank inside each borough, matched by the same SQL, puts Manhattan's busiest hour at 7 p.m. (1.57 million trips), Brooklyn and Queens at 6 p.m., and the Bronx and Staten Island at 8 a.m.

Most trips stay in the pickup borough: 72 percent in Manhattan, 76 percent in Brooklyn, and 86 percent on Staten Island. Queens is the most likely to leave the borough (64 percent stay), which fits a borough that contains both JFK and LaGuardia.

Airport trips are 4.5 million rides, about 8 percent of the quarter. They average **13.4 miles** and **$3.89 per mile**. Other trips average **4.1 miles** and **$6.87 per mile**. The airport fare is spread over a much longer ride, so the per-mile rate is lower even though the trip goes farther. JFK (1.00 million pickups) and LaGuardia (0.96 million) are the two busiest zones. The next zones are Manhattan and Brooklyn neighborhoods such as the East Village, Crown Heights North, and Midtown, where fare per mile sits around $7 to $9.

Recorded tip rates are low: about 5.2 percent of base fare in Manhattan and 1.7 percent in the Bronx. Matched shared rides are about 1 percent of trips. Those figures are what the TLC file stores. A large share of rows have a tip of zero.

## Evidence

- `docs/explain_plan.txt` is the formatted physical plan, including `PushedFilters` and both broadcast joins.
- `docs/metrics.json` is the timings and the aggregates quoted above.
- `docs/screenshots/spark-sql-query-3.png` is the query details view for the borough SQL: scan, filter, broadcast joins, aggregate, 9 seconds, completed.
- `docs/screenshots/spark-sql-list.png` is every SQL execution, all completed, with the 9s, 5s, 0.1s, and 32s durations.
- `docs/screenshots/spark-jobs.png` is the Jobs page, 42 completed jobs, no failures.
- `week13_pipeline.html` is the executed notebook.
