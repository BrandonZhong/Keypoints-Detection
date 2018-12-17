# coding=gbk
import tensorflow as tf
import numpy as np
import csv
import inference
import math


# 配置神经网络的参数
IMAGE_SIZE = 100
APPAREL_SIZE = 80
APPAREL_PART_SIZE = 60
SHUFFLE_BUFFER_SIZE = 50000
LEARNING_RATE_BASE = 0.000002
LEARNING_RATE_DECAY = 0.999
REGULARAZTION_RATE = 0.000001
TRAINING_STEPS = 2
MOVING_AVERAGE_DECAY = 0.99
apparel = 'trousers'
# 模型保存的路径和文件名
MODEL_SAVE_PATH = './model/' + apparel
MODEL_NAME = 'model.ckpt'
keypoints_items = {
	'blouse': {
		'indices': [0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14],
		'begin': 0,
		'end': 5335,
		'batch_size': 55,
	},
	'dress': {
		'indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 17, 18],
		'begin': 5335,
		'end': 10883,
		'batch_size': 73,
	},
	'outwear': {
		'indices': [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
		'begin': 10883,
		'end': 16336,
		'batch_size': 41,
	},
	'skirt': {
		'indices': [15, 16, 17, 18],
		'begin': 16336,
		'end': 21913,
		'batch_size': 39,
	},
	'trousers': {
		'indices': [15, 16, 19, 20, 21, 22, 23],
		'begin': 21913,
		'end': 27223,
		'batch_size': 59
	},
}
keypoints_items = keypoints_items[apparel]
BATCH_SIZE = keypoints_items['batch_size']
begin = keypoints_items['begin']
end = keypoints_items['end']
keypoints_indices = keypoints_items['indices']

def preprocess_for_test(image_id, height, width):
	image = tf.read_file('./test/' + image_id)
	image = tf.image.decode_jpeg(image, channels=3)
	# 转换图像张量的类型
	if image.dtype != tf.float32:
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		
	image = tf.image.per_image_standardization(image)
	image = tf.image.rgb_to_grayscale(image)
	image = tf.image.resize_images(image, size=[height, width])
	
	return image

def decode_line(image_id, image_category):
	
	return image_id, preprocess_for_test(image_id, IMAGE_SIZE, IMAGE_SIZE)

def input_pipeline(filenames):
    image_dataset = tf.data.experimental.CsvDataset(
        filenames, [[''], ['']], header=True)
    image_dataset = image_dataset.apply(tf.data.experimental.map_and_batch(decode_line, BATCH_SIZE))
    
    return image_dataset.make_one_shot_iterator()

image_iterator = input_pipeline('./test/' + apparel + '.csv')
image_id_batch, image_batch = image_iterator.get_next()
image_batch = tf.reshape(image_batch, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])

fine_keypoints = tf.reshape(inference.inference(image_batch, apparel, BATCH_SIZE), [-1, len(keypoints_indices), 2])
fine_length = fine_keypoints.shape.as_list()
ones = tf.ones([fine_length[0], len(keypoints_indices), 1])
fine_keypoints = tf.cast(tf.concat([fine_keypoints, ones], axis=2), tf.int32)

header = ['image_id', 'image_category', 'neckline_left', 'neckline_right', 'center_front', 'shoulder_left', 'shoulder_right', 'armpit_left', 'armpit_right', 'waistline_left', 'waistline_right', 'cuff_left_in', 'cuff_left_out', 'cuff_right_in', 'cuff_right_out', 'top_hem_left', 'top_hem_right', 'waistband_left', 'waistband_right', 'hemline_left', 'hemline_right', 'crotch', 'bottom_left_in', 'bottom_left_out', 'bottom_right_in', 'bottom_right_out']

# 初始化 TensorFlow 持久化类。
saver = tf.train.Saver()

with tf.Session() as sess:
	# tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名。
	ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	# 加载模型
	saver.restore(sess, ckpt.model_checkpoint_path)
	# 通过文件名得到模型保存时迭代的轮数。
	step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
	print('Restore the result of {} training steps, and begin testing.'.format(step))
	
	batch_num = math.ceil((end-begin) / BATCH_SIZE)
	separator = '_'
	with open('./test/' + apparel + '_prediction.csv', 'w', newline='') as wf:
		writer = csv.writer(wf)
		writer.writerow(header)
		for i in range(batch_num):
			images_id, keypoints = sess.run([image_id_batch, fine_keypoints])
			image_keypoints = []
			for j in range(BATCH_SIZE):
				image_keypoints.append([str(images_id[j], encoding='utf-8'), apparel])
				for k in range(24):
					index = 0
					if k in keypoints_indices:
						image_keypoints[j].append(separator.join(str(x)
							for x in keypoints[j][index]))
						index = index + 1
					else:
						image_keypoints[j].append('-1_-1_-1')
			writer.writerows(image_keypoints)
			print('test：' + str(i+1) + '/' + str(batch_num))
