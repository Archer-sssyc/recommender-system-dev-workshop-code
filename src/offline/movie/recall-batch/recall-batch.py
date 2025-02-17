# importing libraries
import argparse
import logging
import os
import pickle

import boto3
import faiss
import pandas as pd
from tqdm import tqdm

import service_impl

########################################
# 从s3同步数据
########################################

def sync_s3(file_name_list, s3_folder, local_folder):
    for f in file_name_list:
        print("file preparation: download src key {} to dst key {}".format(os.path.join(
            s3_folder, f), os.path.join(local_folder, f)))
        s3client.download_file(bucket, os.path.join(
            s3_folder, f), os.path.join(local_folder, f))


def write_to_s3(filename, bucket, key):
    print("upload s3://{}/{}".format(bucket, key))
    with open(filename, 'rb') as f:  # Read in binary mode
        # return s3client.upload_fileobj(f, bucket, key)
        return s3client.put_object(
            ACL='bucket-owner-full-control',
            Bucket=bucket,
            Key=key,
            Body=f
        )


def write_str_to_s3(content, bucket, key):
    print("write s3://{}/{}, content={}".format(bucket, key, content))
    s3client.put_object(Body=str(content).encode("utf8"), Bucket=bucket, Key=key, ACL='bucket-owner-full-control')


parser = argparse.ArgumentParser(description="app inputs and outputs")
parser.add_argument("--bucket", type=str, help="s3 bucket")
parser.add_argument("--prefix", type=str, help="s3 input key prefix")
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

s3client = boto3.client('s3')

local_folder = 'info'
if not os.path.exists(local_folder):
    os.makedirs(local_folder)
# 行为数据加载
file_name_list = ['action.csv']
s3_folder = '{}/system/action-data/'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)
# ub数据加载
file_name_list = ['ub_item_vector.index', 'embed_raw_item_mapping.pickle']
s3_folder = '{}/feature/action/'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)
# 用户画像数据加载
file_name_list = ['portrait.pickle']
s3_folder = '{}/feature/recommend-list/portrait'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)
# 倒排列表的pickle文件
file_name_list = ['card_id_card_property_dict.pickle',
                #   'card_sex_card_ids_dict.pickle',
                  'card_user_card_ids_dict.pickle',
                #   'card_age_card_ids_dict.pickle',
                #   'card_country_card_ids_dict.pickle',
                  'card_name_card_ids_dict.pickle',
                  'card_artist_card_ids_dict.pickle']
s3_folder = '{}/feature/content/inverted-list/'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)

file_name_list = ['recall_config.pickle']
s3_folder = '{}/model/recall'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)

# 加载pickle文件
file_to_load = open("info/card_id_card_property_dict.pickle", "rb")
dict_id_content = pickle.load(file_to_load)
print("length of card_id v.s. card_property {}".format(len(dict_id_content)))

# file_to_load = open("info/card_sex_card_ids_dict.pickle", "rb")
# dict_sex_id = pickle.load(file_to_load)
# print("length of card_sex v.s. card_ids {}".format(len(dict_sex_id)))

file_to_load = open("info/card_user_card_ids_dict.pickle", "rb")
dict_user_id = pickle.load(file_to_load)
print("length of card_dicrector v.s. card_ids {}".format(len(dict_user_id)))

# file_to_load = open("info/card_age_card_ids_dict.pickle", "rb")
# dict_age_id = pickle.load(file_to_load)
# print("length of card_age v.s. card_ids {}".format(len(dict_age_id)))

# file_to_load = open("info/card_country_card_ids_dict.pickle", "rb")
# dict_country_id = pickle.load(file_to_load)
# print("length of card_country v.s. card_ids {}".format(len(dict_country_id)))

file_to_load = open("info/card_name_card_ids_dict.pickle", "rb")
dict_name_id = pickle.load(file_to_load)
print("length of card_name v.s. card_ids {}".format(len(dict_name_id)))

file_to_load = open("info/card_artist_card_ids_dict.pickle", "rb")
dict_artist_id = pickle.load(file_to_load)
print("length of card_artist v.s. card_ids {}".format(len(dict_artist_id)))

file_to_load = open("info/recall_config.pickle", "rb")
recall_config = pickle.load(file_to_load)
print("config recall")

ub_faiss_index = faiss.read_index('info/ub_item_vector.index')

file_to_load = open("info/embed_raw_item_mapping.pickle", "rb")
ub_idx_mapping = pickle.load(file_to_load)
print("length of item mapping {}".format(len(ub_idx_mapping)))

file_to_load = open("info/portrait.pickle", "rb")
user_portrait = pickle.load(file_to_load)
print("length of user_portrait {}".format(len(user_portrait)))

# 配置参数
config_dict = {}
recall_wrap = {}
recall_wrap['content'] = dict_id_content
recall_wrap['dict_wrap'] = {}
# recall_wrap['dict_wrap']['c_singer_sex'] = dict_sex_id
recall_wrap['dict_wrap']['c_singer_user_id'] = dict_user_id
# recall_wrap['dict_wrap']['c_singer_age'] = dict_age_id
# recall_wrap['dict_wrap']['c_singer_country'] = dict_country_id
recall_wrap['dict_wrap']['c_song_name'] = dict_name_id
recall_wrap['dict_wrap']['c_song_artist'] = dict_artist_id
recall_wrap['config'] = recall_config
config_dict['recall_wrap'] = recall_wrap
recall_wrap['ub_index'] = ub_faiss_index
recall_wrap['ub_idx_mapping'] = ub_idx_mapping
# 加载所有人的数据

action_data_pddf = pd.read_csv('info/action.csv').rename(columns={'label': 'rating', 'action_user_id': 'user_id', 'c_id': 'item_id'})

print("load {} action data".format(len(action_data_pddf)))
# 初始化recall结果
recall_batch_result = {}
# print(config_dict)
recall_batch_function = service_impl.ServiceImpl()
for reviewerID, hist in tqdm(
        action_data_pddf[action_data_pddf['rating'] == 1].groupby('user_id')):
    pos_list = hist['item_id'].tolist()
    if str(reviewerID) not in user_portrait:
        logging.warning("Cannot find {} in user_portrait".format(reviewerID))
        continue
    config_dict['user_portrait'] = user_portrait[str(reviewerID)]
    # user_click_records[reviewerID] = pos_list
    recall_batch_result[str(reviewerID)] = recall_batch_function.merge_recall_result([str(elem) for elem in pos_list],
                                                                                     **config_dict)
# 存储recall的结果
file_name = "info/recall_batch_result.pickle"
out_file = open(file_name, 'wb')
pickle.dump(recall_batch_result, out_file)
out_file.close()

write_to_s3(file_name,
            bucket,
            '{}/feature/recommend-list/movie/recall_batch_result.pickle'.format(prefix))
