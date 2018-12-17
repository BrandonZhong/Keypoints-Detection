# coding=gbk
import tensorflow as tf

NUM_CHANNELS = 1
IMAGE_SIZE = 100
APPAREL_SIZE = 80
APPAREL_PART_SIZE = 60

BBOXES_CONV1_SIZE = 5
BBOXES_CONV1_DEEP = 20

BBOXES_CONV2_SIZE = 5
BBOXES_CONV2_DEEP = 40

BBOXES_CONV3_SIZE = 3
BBOXES_CONV3_DEEP = 60

BBOXES_CONV4_SIZE = 3
BBOXES_CONV4_DEEP = 80

BBOXES_CONV5_SIZE = 3
BBOXES_CONV5_DEEP = 100

BBOXES_FC_SIZE = 4

def bounding_box(image):
	# 第一层卷积层，输入100*100*1，输出96*96*20
	with tf.variable_scope('bboxes-layer1-conv1'):
		conv1_weights = tf.get_variable('weight',
			[BBOXES_CONV1_SIZE, BBOXES_CONV1_SIZE, NUM_CHANNELS, BBOXES_CONV1_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.2))
		conv1_biases = tf.get_variable('bias', [BBOXES_CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv1 = tf.nn.conv2d(image, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
	# 第二层池化层，输入96*96*20，输出48*48*20	
	with tf.name_scope('bboxes-layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第三层卷积层，输入48*48*20，输出44*44*40
	with tf.variable_scope('bboxes-layer3-conv2'):
		conv2_weights = tf.get_variable('weight',
			[BBOXES_CONV2_SIZE, BBOXES_CONV2_SIZE, BBOXES_CONV1_DEEP, BBOXES_CONV2_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.2))
		conv2_biases = tf.get_variable('bias', [BBOXES_CONV2_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
	# 第四层池化层，输入44*44*40，输出22*22*40
	with tf.name_scope('bboxes-layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第五层卷积层，输入22*22*40，输出20*20*60
	with tf.variable_scope('bboxes-layer5-conv3'):
		conv3_weights = tf.get_variable('weight',
			[BBOXES_CONV3_SIZE, BBOXES_CONV3_SIZE, BBOXES_CONV2_DEEP, BBOXES_CONV3_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.2))
		conv3_biases = tf.get_variable('bias', [BBOXES_CONV3_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
	# 第六层池化层，输入20*20*60，输出10*10*60
	with tf.name_scope('bboxes-layer6-pool3'):
		pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第七层卷积层，输入10*10*60，输出8*8*60
	with tf.variable_scope('bboxes-layer7-conv4'):
		conv4_weights = tf.get_variable('weight',
			[BBOXES_CONV4_SIZE, BBOXES_CONV4_SIZE, BBOXES_CONV3_DEEP, BBOXES_CONV4_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.2))
		conv4_biases = tf.get_variable('bias', [BBOXES_CONV4_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
	# 第八层池化层，输入8*8*60，输出4*4*80
	with tf.name_scope('bboxes-layer8-pool4'):
		pool4 = tf.nn.max_pool(relu4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第九层卷积层，输入4*4*80，输出2*2*100
	with tf.variable_scope('bboxes-layer9-conv5'):
		conv5_weights = tf.get_variable('weight',
			[BBOXES_CONV5_SIZE, BBOXES_CONV5_SIZE, BBOXES_CONV4_DEEP, BBOXES_CONV5_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.2))
		conv5_biases = tf.get_variable('bias', [BBOXES_CONV5_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv5 = tf.nn.conv2d(pool4, conv5_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu5 = tf.nn.relu(tf.nn.bias_add(conv5, conv5_biases))
	# 将卷积层的输出转化为全连接层的输入格式
	nodes = 400
	reshaped = tf.reshape(relu5, [-1, nodes])
	# 第十层全连接层，输入2*2*100，输出1*4
	with tf.variable_scope('bboxes-layer10-fc'):
		fc_weights = tf.get_variable('weight',
			[nodes, BBOXES_FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.2))
		fc_biases = tf.get_variable('bias', [BBOXES_FC_SIZE], initializer=tf.constant_initializer(0.0))
		
		logits = tf.math.square(tf.matmul(reshaped, fc_weights) + fc_biases)
	return logits

CKEYS_CONV1_SIZE = 5
CKEYS_CONV1_DEEP = 20

CKEYS_CONV2_SIZE = 5
CKEYS_CONV2_DEEP = 40

CKEYS_CONV3_SIZE = 3
CKEYS_CONV3_DEEP = 60

CKEYS_CONV4_SIZE = 3
CKEYS_CONV4_DEEP = 80

def coarse_keypoints_detection(image, ckeys_fc_size):
	# 第一层卷积层，输入60*60*1，输出56*56*20
	with tf.variable_scope('ckeys-layer1-conv1'):
		conv1_weights = tf.get_variable(
			'weight', [CKEYS_CONV1_SIZE, CKEYS_CONV1_SIZE, NUM_CHANNELS, CKEYS_CONV1_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable('bias', [CKEYS_CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv1 = tf.nn.conv2d(image, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
	# 第二层池化层，输入56*56*20，输出28*28*20	
	with tf.name_scope('ckeys-layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第三层卷积层，输入28*28*20，输出24*24*40
	with tf.variable_scope('ckeys-layer3-conv2'):
		conv2_weights = tf.get_variable(
			'weight', [CKEYS_CONV2_SIZE, CKEYS_CONV2_SIZE, CKEYS_CONV1_DEEP, CKEYS_CONV2_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable('bias', [CKEYS_CONV2_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
	# 第四层池化层，输入24*24*40，输出12*12*40
	with tf.name_scope('ckeys-layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第五层卷积层，输入12*12*40，输出10*10*60
	with tf.variable_scope('ckeys-layer5-conv3'):
		conv3_weights = tf.get_variable(
			'weight', [CKEYS_CONV3_SIZE, CKEYS_CONV3_SIZE, CKEYS_CONV2_DEEP, CKEYS_CONV3_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable('bias', [CKEYS_CONV3_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
	# 第六层池化层，输入10*10*60，输出5*5*60
	with tf.name_scope('ckeys-layer6-pool3'):
		pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第七层卷积层，输入5*5*60，输出3*3*80
	with tf.variable_scope('ckeys-layer7-conv4'):
		conv4_weights = tf.get_variable(
			'weight', [CKEYS_CONV4_SIZE, CKEYS_CONV4_SIZE, CKEYS_CONV3_DEEP, CKEYS_CONV4_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable('bias', [CKEYS_CONV4_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
	# 将卷积层的输出转化为全连接层的输入格式
	pool_shape = relu4.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	reshaped = tf.reshape(relu4, [pool_shape[0], nodes])
	# 第十层全连接层，输入2*2*100，输出1*4
	with tf.variable_scope('ckeys-layer8-fc'):
		fc_weights = tf.get_variable(
			'weight', [nodes, ckeys_fc_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc_biases = tf.get_variable('bias', [ckeys_fc_size], initializer=tf.constant_initializer(0.1))
		
		logits = tf.matmul(reshaped, fc_weights) + fc_biases
	
	logits = tf.math.square(logits)
	return logits

FKEYS_CONV1_SIZE = 5
FKEYS_CONV1_DEEP = 20

FKEYS_CONV2_SIZE = 3
FKEYS_CONV2_DEEP = 40

FKEYS_CONV3_SIZE = 3
FKEYS_CONV3_DEEP = 60

FKEYS_CONV4_SIZE = 2
FKEYS_CONV4_DEEP = 80

def fine_keypoints_detection(image, fkeys_fc_size, vscope):
	# 第一层卷积层，输入60*60*1，输出56*56*20
	with tf.variable_scope(vscope + '-layer1-conv1'):
		conv1_weights = tf.get_variable(
			'weight', [FKEYS_CONV1_SIZE, FKEYS_CONV1_SIZE, NUM_CHANNELS, FKEYS_CONV1_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv1_biases = tf.get_variable('bias', [FKEYS_CONV1_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv1 = tf.nn.conv2d(image, conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
	# 第二层池化层，输入56*56*20，输出28*28*20	
	with tf.name_scope(vscope + '-layer2-pool1'):
		pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第三层卷积层，输入28*28*20，输出24*24*40
	with tf.variable_scope(vscope + '-layer3-conv2'):
		conv2_weights = tf.get_variable(
			'weight', [FKEYS_CONV2_SIZE, FKEYS_CONV2_SIZE, FKEYS_CONV1_DEEP, FKEYS_CONV2_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv2_biases = tf.get_variable('bias', [FKEYS_CONV2_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
	# 第四层池化层，输入24*24*40，输出12*12*40
	with tf.name_scope(vscope + '-layer4-pool2'):
		pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第五层卷积层，输入12*12*40，输出10*10*60
	with tf.variable_scope(vscope + '-layer5-conv3'):
		conv3_weights = tf.get_variable(
			'weight', [FKEYS_CONV3_SIZE, FKEYS_CONV3_SIZE, FKEYS_CONV2_DEEP, FKEYS_CONV3_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv3_biases = tf.get_variable('bias', [FKEYS_CONV3_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_biases))
	# 第六层池化层，输入10*10*60，输出5*5*60
	with tf.name_scope(vscope + '-layer6-pool3'):
		pool3 = tf.nn.max_pool(relu3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
	# 第七层卷积层，输入5*5*60，输出3*3*80
	with tf.variable_scope(vscope + '-layer7-conv4'):
		conv4_weights = tf.get_variable(
			'weight', [FKEYS_CONV4_SIZE, FKEYS_CONV4_SIZE, FKEYS_CONV3_DEEP, FKEYS_CONV4_DEEP],
			initializer=tf.truncated_normal_initializer(stddev=0.1))
		conv4_biases = tf.get_variable('bias', [FKEYS_CONV4_DEEP], initializer=tf.constant_initializer(0.0))
		
		conv4 = tf.nn.conv2d(pool3, conv4_weights, strides=[1, 1, 1, 1], padding='VALID')
		relu4 = tf.nn.relu(tf.nn.bias_add(conv4, conv4_biases))
	# 将卷积层的输出转化为全连接层的输入格式
	pool_shape = relu4.get_shape().as_list()
	nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
	reshaped = tf.reshape(relu4, [pool_shape[0], nodes])
	# 第十层全连接层，输入2*2*100，输出1*4
	with tf.variable_scope(vscope + '-layer8-fc'):
		fc_weights = tf.get_variable(
			'weight', [nodes, fkeys_fc_size], initializer=tf.truncated_normal_initializer(stddev=0.1))
		fc_biases = tf.get_variable('bias', [fkeys_fc_size], initializer=tf.constant_initializer(0.1))
		
		logits = tf.matmul(reshaped, fc_weights) + fc_biases
	
	return logits

def images_process(image, height, width, keypoints_length, batch_size, bbox):
	length = tf.cast(tf.tile([[width, height]], [batch_size, 1]), tf.float32)
	images = []
	bbox_begin, bbox_size = bbox
	
	for i in range(batch_size):
		img = tf.image.resize_images(tf.slice(image[i], bbox_begin[i], bbox_size[i]), [height, width])
		images.append(img)
	
	offset_width_height = tf.cast(tf.tile(tf.gather(bbox_begin, [1, 0], axis=1), [1, keypoints_length]), tf.float32)
	image_shape = tf.cast(tf.gather(bbox_size, [1, 0], axis=1), tf.float32)
	image_shape_ratio = tf.tile(tf.math.divide(length, image_shape), [1, keypoints_length])
	#keypoints = tf.math.multiply(tf.math.subtract(keypoints, offset_width_height), image_shape_ratio)
	return images, [offset_width_height, image_shape_ratio]


def apparel_parts(images, height, width, coarse_keypoints, batch_size):
	zero_column = tf.zeros([batch_size, 1], dtype=tf.int32)
	minus_one = tf.tile([[-1]], [batch_size, 1])
	enlarge_time = tf.tile([[1.1, 1.1]], [batch_size, 1])
	keypoints_x_y = tf.gather(tf.reshape(coarse_keypoints, [batch_size, -1, 2]), [1, 0], axis=2)
	keypoints_length = keypoints_x_y.shape.as_list()[1]
	keypoints_x_y_transposed = tf.transpose(keypoints_x_y, perm=[0, 2, 1])
	
	keypoints_min = tf.math.reduce_min(keypoints_x_y_transposed, 2)
	max_begin = tf.cast(tf.tile([[APPAREL_SIZE-5, APPAREL_SIZE-5]], [batch_size, 1]), tf.float32)
	part_begin = tf.math.divide(tf.math.minimum(keypoints_min, max_begin), enlarge_time)
	
	keypoints_max = tf.math.reduce_max(keypoints_x_y_transposed, 2)
	five = tf.cast(tf.tile([[5, 5]], [batch_size, 1]), tf.float32)
	keypoints_max = tf.math.maximum(tf.math.add(part_begin, five), keypoints_max)
	max_size = tf.cast(tf.tile([[APPAREL_SIZE, APPAREL_SIZE]], [batch_size, 1]), tf.float32)
	part_size = tf.math.minimum(keypoints_max, max_size) - part_begin
	
	part_begin = tf.concat([tf.cast(part_begin, tf.int32), zero_column], axis=1)
	part_size = tf.concat([tf.cast(part_size, tf.int32), minus_one], axis=1)
	images, offset_and_ratio = images_process(images, height, width, keypoints_length, batch_size, [part_begin, part_size])
	
	return images, offset_and_ratio

def inference(image_batch, apparel, batch_size):
	keypoints_item = {
		'blouse': {
			'neckline': [0, 1, 2, 3, 4, 5],
			'left-sleeve': [6, 7, 10, 11, 14, 15, 16, 17],
			'right-sleeve': [8, 9, 12, 13, 18, 19, 20, 21],
			'hem': [22, 23, 24, 25],
			'length': 13
		},
		'outwear': {
			'neckline': [0, 1, 2, 3],
			'left-sleeve': [4, 5, 8, 9, 16, 17, 18, 19],
			'right-sleeve': [6, 7, 10, 11, 20, 21, 22, 23],
			'hem': [12, 13, 14, 15, 24, 25, 26, 27],
			'length': 14
		},
		'dress': {
			'neckline': [0, 1, 2, 3, 4, 5],
			'left-sleeve': [6, 7, 10, 11, 18, 19, 20, 21],
			'right-sleeve': [8, 9, 12, 13, 22, 23, 24, 25],
			'hem': [14, 15, 16, 17, 26, 27, 28, 29],
			'length': 15
		},
		'skirt': {
			'hem': [0, 1, 2, 3, 4, 5, 6, 7],
			'length': 4
		},
		'trousers': {
			'neckline': [0, 1, 2, 3, 4, 5],
			'hem': [6, 7, 8, 9, 10, 11, 12, 13],
			'length': 7
		}
	}
	keypoints_item = keypoints_item[apparel]
	keypoints_length = keypoints_item['length']
	del keypoints_item['length']
	zero_column = tf.zeros([batch_size, 1], dtype=tf.int32)
	minus_one = tf.tile([[-1]], [batch_size, 1])
	enlarge_time = tf.tile([[1.1, 1.1]], [batch_size, 1])
	# 预测Bounding Box
	bbox = bounding_box(image_batch)
	
	# 根据Bounding Box求出[offset_height, offset_width]和[target_height, target_width]
	half_length = tf.tile([[IMAGE_SIZE/2, IMAGE_SIZE/2]], [batch_size, 1])
	max_length = tf.cast(tf.tile([[IMAGE_SIZE, IMAGE_SIZE]], [batch_size, 1]), tf.float32)

	bbox_begin = tf.math.minimum(tf.math.divide(bbox[:, 0:2], enlarge_time), half_length)
	bbox_size = tf.math.multiply(bbox[:, 2:4], enlarge_time)
	max_size = tf.math.subtract(max_length, bbox_begin)
	bbox_size = tf.math.maximum(tf.math.minimum(bbox_size, max_size), half_length)
	bbox_begin = tf.concat([tf.cast(bbox_begin, tf.int32), zero_column], axis=1)
	bbox_size = tf.concat([tf.cast(bbox_size, tf.int32), minus_one], axis=1)
	# 根据[offset_height, offset_width]和[target_height, target_width]切割图片
	images, bbox_offset_ratio = images_process(image_batch, APPAREL_SIZE, APPAREL_SIZE,
		keypoints_length, batch_size, [bbox_begin, bbox_size])
	bbox_offset_width_height, bbox_image_shape_ratio = bbox_offset_ratio
	# 粗略地预测keypoints
	coarse_keypoints = coarse_keypoints_detection(images, 2 * keypoints_length)
	parts_fine_keypoints = None
	for vscope, indices in keypoints_item.items():
		# 根据粗略预测出的keypoints把图片切割成衣领、左右衣袖和下摆四个部分
		part_coarse_keypoints = tf.gather(coarse_keypoints, indices, axis=1)
		part_images, part_offset_ratio = apparel_parts(images, APPAREL_PART_SIZE, APPAREL_PART_SIZE, part_coarse_keypoints, batch_size)
		part_offset_width_height, part_image_shape_ratio = part_offset_ratio
		# 每个部分单独进行预测，得到精确预测的keypoints
		part_fine_keypoints = fine_keypoints_detection(part_images, len(indices), vscope)
		part_fine_keypoints = tf.math.add(tf.math.divide(part_fine_keypoints, part_image_shape_ratio), part_offset_width_height)
		if parts_fine_keypoints == None:
			parts_fine_keypoints = part_fine_keypoints
		else:
			parts_fine_keypoints = tf.concat([parts_fine_keypoints, part_fine_keypoints], axis=1)
	fine_keypoints = tf.math.add(tf.math.divide(parts_fine_keypoints, bbox_image_shape_ratio), bbox_offset_width_height)
	
	return fine_keypoints
