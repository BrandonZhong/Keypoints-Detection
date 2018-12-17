# coding=gbk
import os
import collections
import tensorflow as tf
import inference as inference

# 配置神经网络的参数
IMAGE_SIZE = 100
SHUFFLE_BUFFER_SIZE = 50000
BATCH_SIZE = 80
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.999
REGULARAZTION_RATE = 0.000001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
# 模型保存的路径和文件名
apparel = 'outwear'
MODEL_SAVE_PATH = './model/' + apparel
MODEL_NAME = 'model.ckpt'

tf.reset_default_graph()

record_length = {
	'blouse': 15,
	'outwear': 16,
	'dress': 17,
	'skirt': 6,
	'trousers': 9
}
record_length = record_length[apparel]
record_defaults = [[''] for i in range(record_length)]
keypoints_length = record_length - 2
keypoints_indices = []
for i in range(keypoints_length):
	keypoints_indices = keypoints_indices + [3*i, 3*i + 1]

def preprocess_for_train(image, height, width, keypoints):
	# 转换图像张量的类型
	if image.dtype != tf.float32:
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	
	image_shape = tf.shape(image)
	image_shape = tf.gather(image_shape, [1, 0]) # image_shape = [width, height]
	image_shape_ratio = tf.math.divide([width, height], image_shape)
	image_shape_ratio = tf.cast(tf.tile(image_shape_ratio, [keypoints_length]), tf.float32)
	keypoints = tf.math.multiply(keypoints, image_shape_ratio)
	image = tf.image.per_image_standardization(image)
	image = tf.image.rgb_to_grayscale(image)
	# 将随机截取的图像调整为神经网络输入层的大小。大小调整的算法是随机选择的。
	image = tf.image.resize_images(image, size=[height, width])
	
	return image, keypoints
	

# Define how the lines of the file should be parsed
def decode_line(image_id, image_category, *keypoints):
	image = tf.read_file('./train/' + image_id)
	image = tf.image.decode_jpeg(image, channels=3)
	
	keypoints = tf.string_split(keypoints, '_')
	keypoints = keypoints.values
	keypoints = tf.string_to_number(keypoints, out_type=tf.float32)
	keypoints = tf.gather(keypoints, keypoints_indices)
	
	return preprocess_for_train(image, IMAGE_SIZE, IMAGE_SIZE, keypoints)

def input_pipeline(filenames):
    image_dataset = tf.data.experimental.CsvDataset(filenames, record_defaults, header=True)
    image_dataset = image_dataset.apply(tf.data.experimental.shuffle_and_repeat(SHUFFLE_BUFFER_SIZE, count=-1))
    image_dataset = image_dataset.apply(tf.data.experimental.map_and_batch(decode_line, BATCH_SIZE))
    
    return image_dataset.make_one_shot_iterator()

image_iterator = input_pipeline('./train/' + apparel + '.csv')
image_batch, keypoints_batch = image_iterator.get_next()
image_batch = tf.reshape(image_batch, [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, -1])
    
print('数据已全转换成batch。')

# 定义神经网络的结构以及优化过程。image_batch 可以作为输入提供给神经网络的输入层。
# label_batch 则提供了输入 batch 中样例的正确答案
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
#logits = inference.inference(image_batch, True, None, len(keypoints_indices))
fine_keypoints = inference.inference(image_batch, apparel, BATCH_SIZE)
global_step = tf.Variable(0, trainable=False)

# 定义损失函数、准确率、学习率、滑动平均操作以及训练过程
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())

cross_entropy_mean = tf.reduce_mean(tf.square(tf.math.subtract(fine_keypoints, keypoints_batch)))
loss = cross_entropy_mean# + tf.add_n(tf.get_collection('losses'))

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 5000 / BATCH_SIZE, LEARNING_RATE_DECAY)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_step, variable_averages_op]):
	train_op = tf.no_op(name='train')
    
# 初始化 TensorFlow 持久化类。
saver = tf.train.Saver()

# 初始化变量
init_op = tf.global_variables_initializer() # 坑点：不能用tf.global_variables_initializer()

with tf.Session() as sess:
	# tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名。
	ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	if ckpt and ckpt.model_checkpoint_path:
		# 加载模型
		saver.restore(sess, ckpt.model_checkpoint_path)
		# 通过文件名得到模型保存时迭代的轮数。
		step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		print('Restore the result of {} training steps, and continue training.'.format(step))
	else:
		sess.run(init_op)
		print('No checkpoint file found, begin training.')
    
	error_time = 0
	# 在训练过程中不再测试模型在验证数据上的表现，验证和测试的过程将会有一个独立的程序完成。
	for i in range(2):
		_, loss_value, step = sess.run([train_op, loss, global_step])
		print('After {} training steps, loss on training batch is {}.'.format(step, loss_value))
		
		if step % 50 == 0:
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
			print('==========Model has been saved==========')
			
