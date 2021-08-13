import pickle

recall_config = {}
# 可配置的参数
# 每种方法召回的物品数目
recall_config['mt_topn'] = {}
recall_config['mt_topn']['c_singer_sex'] = 10
recall_config['mt_topn']['c_singer_user_id'] = 10
recall_config['mt_topn']['c_singer_age'] = 10
recall_config['mt_topn']['c_singer_country'] = 10
recall_config['mt_topn']['c_song_name'] = 10
recall_config['mt_topn']['c_song_artist'] = 10
recall_config['mt_topn']['review'] = 10
recall_config['mt_topn']['photo'] = 10
recall_config['mt_topn']['portrait_c_song_name'] = 10
recall_config['mt_topn']['portrait_c_song_artist'] = 10
# recall_config['mt_topn']['portrait_actor'] = 10
# recall_config['mt_topn']['portrait_language'] = 10
recall_config['mt_topn']['portrait_ub'] = 10

# 可学习的参数
# 每种产生rank的方法的打分系数，未来通过反馈的ctr进行学习
recall_config['pos_weights'] = {}
recall_config['pos_weights']['c_singer_sex'] = {}
recall_config['pos_weights']['c_singer_sex']['w'] = 0.5
recall_config['pos_weights']['c_singer_sex']['b'] = 0.2
recall_config['pos_weights']['c_singer_user_id'] = {}
recall_config['pos_weights']['c_singer_user_id']['w'] = 0.5
recall_config['pos_weights']['c_singer_user_id']['b'] = 0.2
recall_config['pos_weights']['c_singer_age'] = {}
recall_config['pos_weights']['c_singer_age']['w'] = 0.5
recall_config['pos_weights']['c_singer_age']['b'] = 0.2
recall_config['pos_weights']['c_singer_country'] = {}
recall_config['pos_weights']['c_singer_country']['w'] = 0.5
recall_config['pos_weights']['c_singer_country']['b'] = 0.2
recall_config['pos_weights']['c_song_name'] = {}
recall_config['pos_weights']['c_song_name']['w'] = 0.5
recall_config['pos_weights']['c_song_name']['b'] = 0.2
recall_config['pos_weights']['c_song_artist'] = {}
recall_config['pos_weights']['c_song_artist']['w'] = 0.5
recall_config['pos_weights']['c_song_artist']['b'] = 0.2
recall_config['pos_weights']['portrait_c_song_name'] = {}
recall_config['pos_weights']['portrait_c_song_name']['w'] = 0.5
recall_config['pos_weights']['portrait_c_song_name']['b'] = 0.2
recall_config['pos_weights']['portrait_c_song_artist'] = {}
recall_config['pos_weights']['portrait_c_song_artist']['w'] = 0.5
recall_config['pos_weights']['portrait_c_song_artist']['b'] = 0.2
# recall_config['pos_weights']['portrait_actor'] = {}
# recall_config['pos_weights']['portrait_actor']['w'] = 0.5
# recall_config['pos_weights']['portrait_actor']['b'] = 0.2
# recall_config['pos_weights']['portrait_language'] = {}
# recall_config['pos_weights']['portrait_language']['w'] = 0.5
# recall_config['pos_weights']['portrait_language']['b'] = 0.2

# 每种方法的权重, 未来根据反馈的ctr更新：加权平均
method_weights = {}
method_weights['c_singer_sex'] = 1.0
method_weights['c_singer_user_id'] = 1.0
method_weights['c_singer_age'] = 1.0
method_weights['c_singer_country'] = 1.0
method_weights['c_song_name'] = 1.0
method_weights['c_song_artist'] = 1.0
method_weights['portrait_c_song_name'] = 1.0
method_weights['portrait_c_song_artist'] = 1.0
# method_weights['portrait_actor'] = 1.0
# method_weights['portrait_language'] = 1.0
method_weights['portrait_ub'] = 1.0
recall_config['mt_weights'] = method_weights

# 不同策略的方法
popularity_method_list = ['c_song_name', 'c_song_artist']
portrait_method_list = ['c_song_name', 'c_song_artist']
recall_config['pop_mt_list'] = popularity_method_list
recall_config['portrait_mt_list'] = portrait_method_list

file_name = 'recall_config.pickle'
output_file = open(file_name, 'wb')
pickle.dump(recall_config, output_file)
output_file.close()