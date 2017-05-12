from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import urllib2
import json
from os import listdir
import tensorflow as tf, sys
from tensorflow.python.framework import ops
import time

'''
	1. determine if the given user is a pet owner
		
		a. predict all images by inception
		b. look at the timeline interval to determine if the user is pet owner
'''
# loading graph and model
# load label file  

label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')  

def read_img_from_disk(data_dir, username):
	"""Reads a dir and list all files, read its label
	Args:
		data_dir: where user's timeline photos are, max 100 photos per timeline
	Returns:
		images
    """
    # /public/lchi3/Pet/dog_data
	data_dir = data_dir + '/' + username + '/pics/'
	file_list = listdir(data_dir)
	filenames = []

	for file in file_list:
		if not 'DS_Store' in file:
			filenames.append(data_dir + file)
			
	return filenames

def predict_timeline(images):
	''' Classify each image of the user timeline

	Args: 
		filename queue that represents a user's timeline images
	
	Returns:
		A map of classification result of each image, 
		and the timestamp associated with the image

	'''
	# images = read_img_from_disk('timeline/' + username + '/pics')
	labels = []
	with tf.Session() as sess:
		# Feed the image_data as input to the graph and get first prediction
		# softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		for image_path in images:
			image_data = tf.gfile.FastGFile(image_path, 'rb').read()
			prediction = sess.run(softmax_tensor, \
	             {'DecodeJpeg/contents:0': image_data})

			# Sort to show labels of first prediction in order of confidence
			top_k = prediction[0].argsort()[-len(prediction[0]):][::-1]
			# /public/lchi3/Pet/uid_pid.jpg	
			timestamp = image_path.split('_')[0]
			labels.append({'timestamp': timestamp, 'class': label_lines[top_k[0]]})

	return labels

with open('xxxxxx.json', 'r') as src:
	
	for user_str in src.read().split("\n"):
		start = time.time() 
		
		user_obj = json.loads(user_obj)
		username = user_obj['username']
		
		print ('predicting for user: ' + username + '...')
		images = read_img_from_disk(username)
		labels = predict_timeline(images)
		
		user_obj['pet_labels'] = labels
		output.write(json.dumps(user_obj) + "\n")
		print ('predicting done in ' + str(time.time() - start) + 's')


