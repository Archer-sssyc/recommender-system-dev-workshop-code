import argparse
import json
import time

import boto3
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, size, window, instr, lower


def list_s3_by_prefix(bucket, prefix, filter_func=None):
    print(f"list_s3_by_prefix bucket: {bucket}, prefix: {prefix}")
    s3_bucket = boto3.resource('s3').Bucket(bucket)
    if filter_func is None:
        key_list = [s.key for s in s3_bucket.objects.filter(Prefix=prefix)]
    else:
        key_list = [s.key for s in s3_bucket.objects.filter(
            Prefix=prefix) if filter_func(s.key)]

    print("list_s3_by_prefix return:", key_list)
    return key_list


def s3_copy(bucket, from_key, to_key):
    s3_bucket = boto3.resource('s3').Bucket(bucket)
    copy_source = {
        'Bucket': bucket,
        'Key': from_key
    }
    s3_bucket.copy(copy_source, to_key)
    print("copied s3://{}/{} to s3://{}/{}".format(bucket, from_key, bucket, to_key))


def s3_upload(file, bucket, s3_key):
    s3_bucket = boto3.resource('s3').Bucket(bucket)
    s3_bucket.Object(s3_key).upload_file(file)
    print("uploaded file {} to s3://{}/{}".format(file, bucket, s3_key))


parser = argparse.ArgumentParser(description="app inputs and outputs")
parser.add_argument("--bucket", type=str, help="s3 bucket")
parser.add_argument("--prefix", type=str,
                    help="s3 input key prefix")

parser.add_argument("--region", type=str, help="aws region")
args, _ = parser.parse_known_args()
print("args:", args)

if args.region:
    print("region:", args.region)
    boto3.setup_default_session(region_name=args.region)

bucket = args.bucket
prefix = args.prefix
if prefix.endswith("/"):
    prefix = prefix[:-1]

print(f"bucket:{bucket}, prefix:{prefix}")

# input_prefix=recommender-system-news-open-toutiao/system/item-data/raw-input/
# output_prefix=recommender-system-news-open-toutiao/system/item-data/emr-out/

item_input_file = "s3://{}/{}/system/ingest-data/item/".format(bucket, prefix)
action_input_file = "s3://{}/{}/system/ingest-data/action/".format(bucket, prefix)
user_input_file = "s3://{}/{}/system/ingest-data/user/".format(bucket, prefix)

output_file_key = "{}/system/dashboard/dashboard.json".format(prefix)

print("item_input_file:", item_input_file)
print("action_input_file:", action_input_file)
print("user_input_file:", user_input_file)

# item_input_file = '/Users/yonmzn/tmp/item/'
# action_input_file = '/Users/yonmzn/tmp/action/'
# user_input_file = '/Users/yonmzn/tmp/user/'

statistics_dict = {}
WIN_SIZE = "60 minutes"


def item_statistics(df_item_input):
    print("item_statistics enter")
    global statistics_dict
    total_item_count = df_item_input.select("c_id").dropDuplicates(["c_id"]).count()
    statistics_dict["total_item_count"] = total_item_count
    # is_new_count = df_item_input.selectExpr("id", "cast(is_new as int)").groupby("id").min("is_new").groupby(
    #     "min(is_new)").count().collect()
    # for row in is_new_count:
    #     is_new, count = row['min(is_new)'], row['count']
    #     if is_new == 1:
    #         statistics_dict["new_item_count"] = count
    #         break
    print("item_statistics done")


def user_statistics(df_user_input):
    print("user_statistics enter")
    global statistics_dict
    total_user_count = df_user_input.count()
    # anonymous_user_count = df_user_input.where(
    #     (col('user_name') == '') | (instr(lower(col('user_name')), "anonymous") >= 1)).count()
    # register_user_count = total_user_count - anonymous_user_count
    statistics_dict['total_user_count'] = total_user_count
    # statistics_dict['register_user_count'] = register_user_count
    # statistics_dict['anonymous_user_count'] = anonymous_user_count
    print("user_statistics done")


def action_statistics(df_action_input, df_item_input, df_user_input):
    print("action_statistics enter")
    global statistics_dict
    df_item = df_item_input.select(col("c_id"),
                                   col("c_song_artist"),
                                   col("c_song_name")).dropDuplicates(["c_id"])
    df_item.cache()
    if statistics_dict.get("total_item_count"):
        total_item_count = df_item.count()
    else:
        total_item_count = statistics_dict["total_item_count"]
    df_user = df_user_input.select("u_id")
    df_user.cache()
    #recommender_count = df_action_input.where(col('click_source') == '1').count()
    #recommender_click_count = df_action_input.where((col('click_source') == '1') & (col("action_value") == '1')).count()
    #statistics_dict['recommender_click_cover_ratio'] = int((recommender_click_count / recommender_count) * 100) / 100
    df_clicked_action_event = df_action_input.where(col("label") == '1')
    # total_click_item_count = df_clicked_action_event.select('item_id').dropDuplicates(['item_id']).count()
    item_in_action_count = df_action_input.select(col('c_id')).distinct().count()
    statistics_dict['recommender_click_cover_ratio'] = int((item_in_action_count / total_item_count) * 100) / 100
    click_item_in_action_count = df_action_input.where(col("label") == '1').select(col('c_id')).distinct().count()
    statistics_dict['item_click_ratio'] = int((click_item_in_action_count / item_in_action_count) * 100) / 100
    total_click_count = df_clicked_action_event.count()
    statistics_dict['total_click_count'] = total_click_count
    print("total_click_count: ", total_click_count)
    print("begin finding top_users ...")
    df_action_event = df_clicked_action_event. \
        withColumn("timestamp_bigint", col('stat_date').cast('bigint')). \
        withColumn("event_time", col('timestamp_bigint').cast('timestamp')). \
        drop(col('timestamp_bigint')). \
        drop(col('label')). \
        drop(col('stat_date'))
    join_type = "inner"
    df_action_event_full = df_action_event.join(df_item, ['c_id'], join_type) \
        .join(df_user, ['u_id'], join_type)
    # df_action_hour = df_action_event_full.withColumn('date', col('event_time').cast("date")).withColumn('hour', hour(col('event_time')))
    df_action_user_window = df_action_event_full.groupBy(window(col('event_time'), WIN_SIZE), col('u_id')).count()
    df_action_user_window_sort = df_action_user_window.orderBy([col('window').desc(), col('count').desc()])
    user_rows = df_action_user_window_sort.select(col("u_id"), col('count')).take(100)
    top_10_user_ids = []
    top_10_user = []
    for user in user_rows:
        user_id = user['u_id']
        # user_name = user['user_name']
        count = user['count']
        if user_id not in top_10_user_ids:
            top_10_user_ids.append(user_id)
            top_10_user.append({
                'u_id': user_id,
                # "name": user_name,
                "count": int(count)
            })
        if len(top_10_user_ids) >= 10:
            break

    statistics_dict['top_users'] = top_10_user
    print("begin finding top_items ...")
    df_action_item_window = df_action_event_full.groupBy(window(col('event_time'), WIN_SIZE), col('c_id'),
                                                         col('c_song_name')).count()
    df_action_item_window_sort = df_action_item_window.orderBy([col('window').desc(), col('count').desc()])
    item_rows = df_action_item_window_sort.select(col("c_id"), col("c_song_name"), col('count')).take(100)
    top_10_item_ids = []
    top_10_item = []
    for item_row in item_rows:
        item_id = item_row['c_id']
        title = item_row['c_song_name']
        count = item_row['count']
        if item_id not in top_10_item_ids:
            top_10_item_ids.append(item_id)
            top_10_item.append({
                "c_id": item_id,
                "c_song_name": title,
                "count": int(count)
            })
        if len(top_10_item_ids) >= 10:
            break
    statistics_dict['top_items'] = top_10_item
    # print("begin finding click_count_by_source ...")
    # click_count_by_source = []
    # action_source_total_rows = df_action_event_full.groupBy(col('click_source')).count().collect()
    # for row in action_source_total_rows:
    #     click_count_by_source.append(
    #         {
    #             "source": 'recommend' if str(row['click_source']) == '1' else row['click_source'],
    #             "count": row['count']
    #         }
    #     )
    # statistics_dict['click_count_by_source'] = click_count_by_source
    # print("begin finding click_count_by_source_time_window ...")
    # df_action_event_recommender = df_action_event_full.withColumn("is_recommender", col('click_source') == '1')
    # df_action_event_recommender_window = df_action_event_recommender \
    #     .groupBy(window(col('event_time'), WIN_SIZE), col('is_recommender')).count()
    # n_hours = 8
    # start_time = \
    #     df_action_event_recommender_window.select(col("window")['start']).orderBy([col('window.start').desc()]).take(
    #         n_hours)[-1][
    #         'window.start']
    # df_action_recommender_n_hours = df_action_event_recommender_window.where(
    #     col("window")['start'] >= start_time).orderBy(
    #     [col('window')])
    # recommender_n_hours = df_action_recommender_n_hours.collect()
    # clicked_by_recommender = []
    # for row in recommender_n_hours:
    #     start_time = int(row['window']['start'].timestamp())
    #     end_time = int(row['window']['end'].timestamp())
    #     is_recommender = row['is_recommender']
    #     count = row['count']
    #     clicked_by_recommender.append({
    #         "start_time": start_time,
    #         "end_time": end_time,
    #         "is_recommender": is_recommender,
    #         "count": count
    #     })
    # statistics_dict['click_count_recommender_time_window'] = clicked_by_recommender


with SparkSession.builder.appName("Spark App - item preprocessing").getOrCreate() as spark:
    #
    # item data
    #
    df_item_input = spark.read.csv(item_input_file, header=True)
    # df_item_input = df_item_input.selectExpr("split(value, '_!_') as row").where(
    #     size(col("row")) > 12).selectExpr("row[0] as item_id",
    #                                       "row[1] as program_type",
    #                                       "row[2] as title",
    #                                       "row[3] as release_year",
    #                                       "row[4] as director",
    #                                       "row[5] as actor",
    #                                       "row[6] as category_property",
    #                                       "row[7] as language",
    #                                       "row[8] as ticket_num",
    #                                       "row[9] as popularity",
    #                                       "row[10] as score",
    #                                       "row[11] as level",
    #                                       "row[12] as is_new"
    #                                       )
    df_item_input.cache()
    print("df_item_input OK")

    #
    # action data
    #

    df_action_input = spark.read.csv(action_input_file, header=True)
    # df_action_input = df_action_input.selectExpr("split(value, '_!_') as row").where(
    #     size(col("row")) > 5).selectExpr("row[0] as user_id",
    #                                      "row[1] as item_id",
    #                                      "row[2] as timestamp",
    #                                      "cast(row[3] as string) as action_type",
    #                                      "row[4] as action_value",
    #                                      "row[5] as click_source",
    #                                      )
    # df_action_input = df_action_input.withColumn("click_source", lit("1"))
    df_action_input.cache()
    print("df_action_input OK")

    #
    # user data
    #
    df_user_input = spark.read.csv(user_input_file, header=True)
    # df_user_input = df_user_input.selectExpr("split(value, '_!_') as row").where(
    #     size(col("row")) > 4).selectExpr("row[0] as user_id",
    #                                      "row[4] as user_name",
    #                                      ).dropDuplicates(["user_id"])
    print("df_user_input OK")
    df_user_input.cache()

    item_statistics(df_item_input)
    action_statistics(df_action_input, df_item_input, df_user_input)
    user_statistics(df_user_input)

statistics_dict["report_time"] = int(time.time())
print("statistics_dict:", statistics_dict)
file_name = "dashboard.json"
with open(file_name, 'w', encoding='utf8') as out:
    json.dump(statistics_dict, out, indent=1, ensure_ascii=False)

s3_upload(file_name, bucket, output_file_key)
print("Done!")
