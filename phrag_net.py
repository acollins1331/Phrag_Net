import tensorflow as tf # tensorflow module
import argparse
import numpy as np # numpy module
import pandas as pd
import glob as glob
import math
import matplotlib.pyplot as plt
import sklearn
import seaborn
import tifffile as tif
import os # path join
from performance import *
from make_confusion_matrix import *
from heapq import merge
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

FEATURES_COUNT = 9
TARGET_COUNT = 1
batch_size = 100000
learning_rate = 0.0005
loss = 'categorical_crossentropy'
dense_activation = tf.keras.activations.tanh
activation = tf.keras.activations.softmax
optimizer = tf.keras.optimizers.Nadam(lr=learning_rate)

"""""
training = (glob.glob('./Classifier/MS/*.tif'))
mask = (glob.glob('./Classifier/Training/*.tif'))
testing = (glob.glob('./Full/MS/*.tif'))
test_mask = (glob.glob('./Full/Training/*.tif'))
train_X = []
train_y = []
test_X = []
test_y = []
val_X = []
val_y = []

for i in range(len(test_mask)):
	print(i)
	image = tif.imread(testing[i])
	label = tif.imread(test_mask[i])
	#image = image.astype(int)
	#label = label.astype(int)
	#fig = plt.figure()
	#ax0 = fig.add_subplot(1, 2, 1), plt.imshow(image[:, :, :3])
	#ax1 = fig.add_subplot(1, 2, 2), plt.imshow(label)
	#plt.show()
	image = image.reshape(-1, image.shape[-1])
	label = label.flatten()
	data_indices = [i for i, e in enumerate(label) if e != 0]
	try:
		image = image[data_indices, :]
		label = label[data_indices]
		print(np.shape(image))
		print(np.shape(label))
		test_X.append(image)
		test_y.append(label)
		print(image)
	except:
		continue

np.save('./test_values', test_X)
np.save('./testy_values', test_y)

"""
"""""
train_X = np.load('./train_values.npy', allow_pickle=True)
train_y = np.load('./target_values.npy', allow_pickle=True)

temp = np.zeros((0,9))
temp_label = []
val_label = []
phrag_cells = np.zeros((0,9))
native_cells = np.zeros((0,9))
ground_cells = np.zeros((0,9))
road_cells = np.zeros((0,9))
cloud_cells = np.zeros((0,9))
water_cells = np.zeros((0,9))
tree_cells = np.zeros((0,9))
val_cells = np.zeros((0,9))


for i in range(len(train_y)):
	print(i)
	native_indices = []
	phrag_indices = []
	ground_indices = []
	road_indices = []
	cloud_indices = []
	water_indices = []
	tree_indices = []
	native_indices += [i for i, e in enumerate(train_y[i].flatten()) if e == 1]
	phrag_indices += [i for i, e in enumerate(train_y[i].flatten()) if e == 2]
	ground_indices += [i for i, e in enumerate(train_y[i].flatten()) if e == 3]
	road_indices += [i for i, e in enumerate(train_y[i].flatten()) if e == 4]
	cloud_indices += [i for i, e in enumerate(train_y[i].flatten()) if e == 5]
	water_indices += [i for i, e in enumerate(train_y[i].flatten()) if e == 6]
	tree_indices += [i for i, e in enumerate(train_y[i].flatten()) if e == 7]
	image = train_X[i]
	label = train_y[i]
	phrag_cells = np.append(phrag_cells, image[phrag_indices], axis=0)
	native_cells = np.append(native_cells, image[native_indices], axis=0)
	ground_cells = np.append(ground_cells, image[ground_indices], axis=0)
	road_cells = np.append(road_cells, image[road_indices], axis=0)
	cloud_cells = np.append(cloud_cells, image[cloud_indices], axis=0)
	water_cells = np.append(water_cells, image[water_indices], axis=0)
	tree_cells = np.append(tree_cells, image[tree_indices], axis=0)

class_length = 400000
class_length_2 = len(phrag_cells)
val_cells = np.append(val_cells, phrag_cells[class_length:class_length_2, :], axis=0)
val_cells = np.append(val_cells, native_cells[class_length:class_length_2, :], axis=0)
val_cells = np.append(val_cells, ground_cells[class_length:class_length_2, :], axis=0)
val_cells = np.append(val_cells, road_cells[class_length:class_length_2, :], axis=0)
val_cells = np.append(val_cells, cloud_cells[class_length:class_length_2, :], axis=0)
val_cells = np.append(val_cells, water_cells[class_length:class_length_2, :], axis=0)
val_cells = np.append(val_cells, tree_cells[class_length:class_length_2, :], axis=0)
val_label = np.append(val_label, [2 for i in range(len(phrag_cells[class_length:class_length_2, :]))], axis=0)
val_label = np.append(val_label, [1 for i in range(len(native_cells[class_length:class_length_2, :]))], axis=0)
val_label = np.append(val_label, [3 for i in range(len(ground_cells[class_length:class_length_2, :]))], axis=0)
val_label = np.append(val_label, [4 for i in range(len(road_cells[class_length:class_length_2, :]))], axis=0)
val_label = np.append(val_label, [5 for i in range(len(cloud_cells[class_length:class_length_2, :]))], axis=0)
val_label = np.append(val_label, [6 for i in range(len(water_cells[class_length:class_length_2, :]))], axis=0)
val_label = np.append(val_label, [7 for i in range(len(tree_cells[class_length:class_length_2, :]))], axis=0)

phrag_cells = phrag_cells[:class_length, :]
native_cells = native_cells[:class_length, :]
ground_cells = ground_cells[:class_length, :]
road_cells = road_cells[:class_length, :]
cloud_cells = cloud_cells[:class_length, :]
water_cells = water_cells[:class_length, :]
tree_cells = tree_cells[:class_length, :]

temp = np.append(temp, phrag_cells, axis=0)
temp = np.append(temp, native_cells, axis=0)
temp = np.append(temp, ground_cells, axis=0)
temp = np.append(temp, road_cells, axis=0)
temp = np.append(temp, cloud_cells, axis=0)
temp = np.append(temp, water_cells, axis=0)
temp = np.append(temp, tree_cells, axis=0)
temp_label = np.append(temp_label, [2 for i in range(class_length)], axis=0)
temp_label = np.append(temp_label, [1 for i in range(class_length)], axis=0)
temp_label = np.append(temp_label, [3 for i in range(len(ground_cells))], axis=0)
temp_label = np.append(temp_label, [4 for i in range(len(road_cells))], axis=0)
temp_label = np.append(temp_label, [5 for i in range(class_length)], axis=0)
temp_label = np.append(temp_label, [6 for i in range(class_length)], axis=0)
temp_label = np.append(temp_label, [7 for i in range(class_length)], axis=0)
print(np.shape(val_cells))
print(np.shape(val_label))
#np.save('./train_X', temp)
#np.save('./train_y', temp_label)
np.save('./val_X', val_cells)
np.save('./val_y', val_label)
"""
class myOnet(object):
	def __init__(self):
		self.old_best = 500
		self.val_loss = 500
		self.fail_counter = 0

	def validate(self, my_model):
		writer = tf.summary.create_file_writer(logs_path)
		with writer.as_default():
			tf.summary.scalar("Val_Loss", self.val_loss, step=1)
		writer.flush()
		if self.val_loss < self.old_best:
			print(
				str(self.old_best) + ' was the old best val_loss. ' + str(self.val_loss) + ' is the new best val loss!')
			self.old_best = self.val_loss
			my_model.save('./results/val_loss'+ str(loss)+ '_'+ str(learning_rate)+ "_tanh-2.h5", overwrite=True)
			self.fail_counter = 0
		else:
			self.fail_counter += 1
			print("consecutive val fails: " + str(self.fail_counter))
			if self.fail_counter % 1 == 0:
				print("val loss failed to improve 1 epochs in a row")
				print("Current LR: " + str(my_model.optimizer.lr) + "reducing learning rate by 1%")
				tf.keras.backend.set_value(my_model.optimizer.lr, my_model.optimizer.lr * .99)
				print("New LR: " + str(tf.keras.backend.get_value(my_model.optimizer.lr)))

	def get_batch(self, train_X, train_y, i):
		random_int = 0#np.random.randint(0, len(trainX))
		#print(random_int)
		train_batch = train_X
		target_batch = train_y
		train_batch = train_batch[(batch_size*i):(batch_size*i)+batch_size, :]
		target_batch = target_batch[(batch_size*i):(batch_size*i)+batch_size]
		return train_batch, target_batch

	def network(self):

		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Dense(1024, use_bias=True, kernel_initializer='random_uniform', input_dim=FEATURES_COUNT, activation=None))
		model.add(tf.keras.layers.Dense(1024, use_bias=True, kernel_initializer='random_uniform',  activation=dense_activation))
		model.add(tf.keras.layers.Dropout(0.2, input_shape=(1024,)))
		model.add(tf.keras.layers.Dense(512, use_bias=True, kernel_initializer='random_uniform',  activation=dense_activation))
		model.add(tf.keras.layers.Dropout(0.2, input_shape=(512,)))
		model.add(tf.keras.layers.Dense(256, use_bias=True, kernel_initializer='random_uniform',  activation=dense_activation))
		model.add(tf.keras.layers.Dropout(0.2, input_shape=(256,)))
		model.add(tf.keras.layers.Dense(8, activation=activation))
		model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.categorical_accuracy])
		model.summary()
		return model

	def train(self):
		sample_weights = np.ones((100000,8))
		try:
			my_model = tf.keras.models.load_model('./results/iter'+ str(loss)+ '_'+ str(learning_rate)+ "_tanh-2.h5")
			print("loaded chk")
		except:
			print("couldnt load chk")
			my_model = self.network()
		#train model for an epoch
		for epoch in range(1000):
			epoch_length = len(train_X)/batch_size
			writer = tf.summary.create_file_writer(logs_path)

			valloss = []
			print(epoch)
			epoch_mean_loss = []
			if epoch % 5 == 0:
				val_y = np.load('./val_y.npy')
				np.random.set_state(rng_state)
				np.random.shuffle(val_y)
				val_y = tf.keras.utils.to_categorical(val_y, num_classes=8)
				val_history = my_model.evaluate(val_X, val_y, batch_size=batch_size, verbose=1)
				valloss = np.append(valloss, val_history[0])
				self.val_loss = float(np.mean(valloss))
				writer.flush()
				self.validate(my_model)

			for i in range(int(epoch_length)):
				#print("batch: ", i, "/", epoch_length)
				train_batch, target_batch = self.get_batch(train_X, train_y, i)
				target_batch = tf.keras.utils.to_categorical(target_batch, num_classes=8)
				train_history = my_model.train_on_batch(train_batch, target_batch)
				with writer.as_default():
					tf.summary.scalar("Loss", train_history[0], step=i)
				#print("loss: ", train_history[0])
				epoch_mean_loss = np.append(epoch_mean_loss, train_history[0])
			print("Epoch Loss: " + str(np.mean(epoch_mean_loss)))
			my_model.save(('./results/iter'+ str(loss)+ '_'+ str(learning_rate)+ "_tanh-2.h5"), overwrite=True)


	def eval(self):
		my_model = tf.keras.models.load_model('./results/val_loss'+ str(loss)+ '_'+ str(learning_rate)+"_tanh-2.h5")
		cm = np.zeros((8, 8))
		infer_out = my_model.predict(val_X, batch_size=batch_size, verbose=1)
		infer_class = np.argmax(infer_out, axis=1)
		val_y = np.load('./val_y.npy')
		np.random.set_state(rng_state)
		np.random.shuffle(val_y)
		val_class = val_y
		cm+=confusion_matrix(val_class [1:], infer_class[1:], labels=[0, 1, 2, 3, 4, 5, 6, 7])

		categories = ["Unclassified", "Native", "Phrag", "Ground", "Road", "Cloud", "Water", "Tree"]
		df_cm = pd.DataFrame(cm, index=categories, columns=categories)
		pretty_plot_confusion_matrix(df_cm, annot=True, cmap="Blues", fmt='.2f', fz=11,
									 lw=0.5, cbar=False, figsize=[12, 12], show_null_values=0, pred_val_axis='x')
		plt.show()

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-tr', '--train', action='store_true',
						help="train model")
	parser.add_argument('-te', '--test', action='store_true',
						help="test model")
	args = parser.parse_args()
	val_X = np.load('./val_X.npy')
	train_y = np.load('./train_y.npy')
	train_X = np.load('./train_X.npy')
	rng_state = np.random.get_state()
	np.random.shuffle(train_X)
	np.random.set_state(rng_state)
	np.random.shuffle(train_y)
	np.random.set_state(rng_state)
	np.random.shuffle(val_X)

	Onet = myOnet()
	search = 0
while search < 100:
	logs_path = "./logs/" + str(loss) + "_" + str(learning_rate) + "_tanh-2"
	if args.train:
		Onet.train()
	if args.test:
		Onet.eval()
	learning_rate-=.0003
	search+=1