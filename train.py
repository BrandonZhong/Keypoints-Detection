# coding=gbk
import os
import collections
import tensorflow as tf
import inference as inference

# ����������Ĳ���
IMAGE_SIZE = 100
SHUFFLE_BUFFER_SIZE = 50000
BATCH_SIZE = 80
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.999
REGULARAZTION_RATE = 0.000001
TRAINING_STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
# ģ�ͱ����·�����ļ���
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
	# ת��ͼ������������
	if image.dtype != tf.float32:
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
	
	image_shape = tf.shape(image)
	image_shape = tf.gather(image_shape, [1, 0]) # image_shape = [width, height]
	image_shape_ratio = tf.math.divide([width, height], image_shape)
	image_shape_ratio = tf.cast(tf.tile(image_shape_ratio, [keypoints_length]), tf.float32)
	keypoints = tf.math.multiply(keypoints, image_shape_ratio)
	image = tf.image.per_image_standardization(image)
	image = tf.image.rgb_to_grayscale(image)
	# �������ȡ��ͼ�����Ϊ�����������Ĵ�С����С�������㷨�����ѡ��ġ�
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
    
print('������ȫת����batch��')

# ����������Ľṹ�Լ��Ż����̡�image_batch ������Ϊ�����ṩ�������������㡣
# label_batch ���ṩ������ batch ����������ȷ��
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
#logits = inference.inference(image_batch, True, None, len(keypoints_indices))
fine_keypoints = inference.inference(image_batch, apparel, BATCH_SIZE)
global_step = tf.Variable(0, trainable=False)

# ������ʧ������׼ȷ�ʡ�ѧϰ�ʡ�����ƽ�������Լ�ѵ������
variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
variable_averages_op = variable_averages.apply(tf.trainable_variables())

cross_entropy_mean = tf.reduce_mean(tf.square(tf.math.subtract(fine_keypoints, keypoints_batch)))
loss = cross_entropy_mean# + tf.add_n(tf.get_collection('losses'))

learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 5000 / BATCH_SIZE, LEARNING_RATE_DECAY)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
with tf.control_dependencies([train_step, variable_averages_op]):
	train_op = tf.no_op(name='train')
    
# ��ʼ�� TensorFlow �־û��ࡣ
saver = tf.train.Saver()

# ��ʼ������
init_op = tf.global_variables_initializer() # �ӵ㣺������tf.global_variables_initializer()

with tf.Session() as sess:
	# tf.train.get_checkpoint_state������ͨ��checkpoint�ļ��Զ��ҵ�Ŀ¼������ģ�͵��ļ�����
	ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
	if ckpt and ckpt.model_checkpoint_path:
		# ����ģ��
		saver.restore(sess, ckpt.model_checkpoint_path)
		# ͨ���ļ����õ�ģ�ͱ���ʱ������������
		step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
		print('Restore the result of {} training steps, and continue training.'.format(step))
	else:
		sess.run(init_op)
		print('No checkpoint file found, begin training.')
    
	error_time = 0
	# ��ѵ�������в��ٲ���ģ������֤�����ϵı��֣���֤�Ͳ��ԵĹ��̽�����һ�������ĳ�����ɡ�
	for i in range(2):
		_, loss_value, step = sess.run([train_op, loss, global_step])
		print('After {} training steps, loss on training batch is {}.'.format(step, loss_value))
		
		if step % 50 == 0:
			saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
			print('==========Model has been saved==========')
			
