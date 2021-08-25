import argparse
import logging
import os
import pickle
import re

from tqdm import tqdm

tqdm.pandas()

import boto3
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
                    datefmt='%Y-%m-%d:%H:%M:%S',
                    level=logging.INFO)


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


def prepare_df(item_path):
    df = pd.read_csv(item_path)
    df['c_id'] = df['c_id'].values.astype('int64')
    return df


def get_actor(actor_str):
    if not actor_str or str(actor_str).lower() in ['nan', 'nr', '']:
        return [None]
    actor_arr = actor_str.split('|')
    return [item.strip().lower() for item in actor_arr]


def get_category(category_property):
    if not category_property or str(category_property).lower() in ['nan', 'nr', '']:
        return [None]
    if not category_property:
        return [None]
    return [item.strip().lower() for item in category_property.split('|')]


def get_single_item(item):
    if not item or str(item).lower().strip() in ['nan', 'nr', '']:
        return [None]
    return [str(item).lower().strip()]


def item_embed(x, raw_embed_item_mapping, ub_item_embeddings):
    embed_item_idx = raw_embed_item_mapping[str(x)]
    if int(embed_item_idx) < len(ub_item_embeddings):
        #         print(user_portrait[x])
        return ub_item_embeddings[int(embed_item_idx)]
    else:
        return [0] * embed_dim


def item_id_feat(x, i):
    return x[i]


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

s3 = boto3.client('s3')
s3client = s3

local_folder = 'info'
if not os.path.exists(local_folder):
    os.makedirs(local_folder)

# youtubednn模型数据加载
file_name_list = ['raw_embed_item_mapping.pickle',
                  'raw_embed_user_mapping.pickle']
s3_folder = '{}/feature/action/'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)

file_name_list = ['ub_item_embeddings.npy']
s3_folder = '{}/feature/action/'.format(prefix)
ub_item_exists = False
try:
    sync_s3(file_name_list, s3_folder, local_folder)
    ub_item_exists = True
except Exception as e:
    print("run as init load, cannot find ub_item_embeddings.npy")
    print(repr(e))

# 倒排列表的pickle文件
file_name_list = ['card_id_card_property_dict.pickle']
s3_folder = '{}/feature/content/inverted-list/'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)

file_name_list = ['item.csv']
s3_folder = '{}/system/item-data/'.format(prefix)
sync_s3(file_name_list, s3_folder, local_folder)

# 加载pickle文件
file_to_load = open("info/card_id_card_property_dict.pickle", "rb")
dict_id_content = pickle.load(file_to_load)
print("length of card_id v.s. card_property {}".format(len(dict_id_content)))
file_to_load = open("info/raw_embed_item_mapping.pickle", "rb")
raw_embed_item_mapping = pickle.load(file_to_load)
file_to_load = open("info/raw_embed_user_mapping.pickle", "rb")
raw_embed_user_mapping = pickle.load(file_to_load)


#     return pd.Series(f_dict)

def sparse_item_id_feat(x, mt, dict_id_content=dict_id_content):
    result = dict_id_content[str(x)][mt]
    if result[0] is None:
        return None
    else:
        return '|'.join(result)


# 加载模型
# user_embedding_model = load_model('info/user_embeddings.h5', custom_objects)
if ub_item_exists:
    ub_item_embeddings = np.load("info/ub_item_embeddings.npy")
else:
    ub_item_embeddings = []

embed_dim = 32

df = prepare_df("info/item.csv")
card_id_card_property_data = {}
row_cnt = 0
for row in df.iterrows():
    item_row = row[1]
    program_id = str(item_row['c_id'])
    program_dict = {
        # 'c_singer_sex': str(item_row['c_singer_sex']),
        'c_singer_user_id': str(item_row['c_singer_user_id']),
        # 'c_singer_age': str(item_row['c_singer_age']),
        # 'c_singer_country': str(item_row['c_singer_country']),
        'c_song_name': str(item_row['c_song_name']),
        'c_song_artist': str(item_row['c_song_artist'])
    }
    row_content = []
    row_content.append(str(item_row['c_id']))
    # row_content.append(program_dict['c_singer_sex'])
    row_content.append(program_dict['c_singer_user_id'])
    # row_content.append(program_dict['c_singer_age'])
    # row_content.append(program_dict['c_singer_country'])
    row_content.append(program_dict['c_song_name'])
    row_content.append(program_dict['c_song_artist'])
    card_id_card_property_data['row_{}'.format(row_cnt)] = row_content
    row_cnt = row_cnt + 1

raw_data_pddf = pd.DataFrame.from_dict(card_id_card_property_data, orient='index',
                                       columns=['c_id', 'c_singer_user_id', 'c_song_name',
                                                'c_song_artist'])
raw_data_pddf = raw_data_pddf.reset_index(drop=True)

sample_data_pddf = raw_data_pddf

# item id feature - item embedding
print("根据item_id索引itemid_feat（嵌入）")
sample_data_pddf['itemid_feat'] = sample_data_pddf['c_id'].progress_apply(
    lambda x: item_embed(x, raw_embed_item_mapping, ub_item_embeddings))
print("将{}维物品嵌入转化为不同的连续型feature".format(embed_dim))
for i in tqdm(range(embed_dim)):
    sample_data_pddf['item_feature_{}'.format(i)] = sample_data_pddf['itemid_feat'].apply(lambda x: item_id_feat(x, i))
# sparse feature
print("根据item_id对应的content生成离散feature")
popularity_method_list = ['c_singer_user_id', 'c_song_name',
                                                'c_song_artist']
for i, mt in tqdm(enumerate(popularity_method_list)):
    sample_data_pddf['sparse_feature_{}'.format(i)] = sample_data_pddf['c_id'].apply(
        lambda x: sparse_item_id_feat(x, mt))

mk_data = sample_data_pddf
dense_feature_size = embed_dim
sparse_feature_size = 3  #should be changed by fact
for i in range(dense_feature_size):
    mk_data['I{}'.format(i + embed_dim)] = mk_data['item_feature_{}'.format(i)]
for i in range(sparse_feature_size):
    mk_data['C{}'.format(i + 1)] = mk_data['sparse_feature_{}'.format(i)]

mk_sparse_features = ['C' + str(i) for i in range(1, sparse_feature_size + 1)]
mk_dense_features = ['I' + str(i + embed_dim - 1) for i in range(1, dense_feature_size + 1)]
mk_data[mk_sparse_features] = mk_data[mk_sparse_features].fillna('-1', )
mk_data[mk_dense_features] = mk_data[mk_dense_features].fillna(0, )

for feat in mk_sparse_features:
    lbe = LabelEncoder()
    mk_data[feat] = lbe.fit_transform(mk_data[feat])
nms = MinMaxScaler(feature_range=(0, 1))
mk_data[mk_dense_features] = nms.fit_transform(mk_data[mk_dense_features])

card_id_card_feature_data = {}
for row in mk_data.iterrows():
    item_row = row[1]
    #     print(item_row)
    #     break
    program_dict = str(item_row['c_id'])
    row_content = []
    row_content.append(str(item_row['c_id']))
    dense_score = []
    for feat in mk_sparse_features:
        row_content.append(item_row[feat])
    for feat in mk_dense_features:
        row_content.append(item_row[feat])
        dense_score.append(item_row[feat])
    row_content.append(np.mean(dense_score))
    card_id_card_feature_data['row_{}'.format(row_cnt)] = row_content
    row_cnt = row_cnt + 1

col_names = ['c_id'] + mk_sparse_features + mk_dense_features + ['item_feat_mean']
mk_item_feature_pddf = pd.DataFrame.from_dict(card_id_card_feature_data, orient='index', columns=col_names)
mk_item_feature_pddf = mk_item_feature_pddf.reset_index(drop=True)

file_name = 'info/card_id_card_feature_dict.pickle'
mk_item_feature_pddf.to_pickle(file_name)
write_to_s3(file_name, bucket, "{}/feature/content/inverted-list/{}".format(prefix, file_name.split('/')[-1]))
