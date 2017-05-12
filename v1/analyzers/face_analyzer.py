'''
	Facial analysis

	2. face ++ 

		a. locate the user by face groupping
		b. identify user has partner, has child or not

			1). parse all image captions to capture keyword
				i). #mygirdfriend, #myboyfriend, #myhusband, #mywife, #mychildreb etc.
			2). person repeatedly appear in the user's timeline => has partener 
			3). child repeatedly appear in the user's timeline => has child

		c. get the average smile index
''' 
import logging
import time
import json
from os import listdir

from facepp import API
from facepp import File
from pprint import pformat

api = ''
BATCH_ID = ''
output = ''
output_external_usage = ''

MAX_FACESET = 5
DELAY = 5
FILE_THRESHOLD = 30
FACE_THRESHOLD = 5

SESSION_INQUEUE = []
SESSION_FACESET_MAP = {}
FACE_POST_TIME_MAP = {}

REMOVEABLE_FACESETS = []

def init(batch_id, key, secret):
	global api, BATCH_ID, output, output_external_usage
	# constants
	api = API(key, secret)
	BATCH_ID = batch_id
	log_dir = 'logs/0/cat_log/log_batch_' + BATCH_ID + '.log'
	output_dir = 'output/0/cat/output_batch_' + BATCH_ID + '.txt'
	output_external_usage_dir = 'output/0/cat/output_external_usage_batch_' + BATCH_ID + '.txt'
	logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO, filename=log_dir)
	output = open(output_dir, 'a+')
	output_external_usage = open(output_external_usage_dir, 'a+')


def delete_faceset(faceset_id):
	global REMOVEABLE_FACESETS
	# make sure the faceset is deleted
	del_resp = api.faceset.delete(faceset_id=faceset_id)

	while not del_resp['success']:
		logging.info ('BATCH_' + BATCH_ID + ':faceset:' + faceset_id + ' delete failed, retry... ')
		del_resp = api.faceset.delete(faceset_id=faceset_id)
		time.sleep(1)

	if del_resp['success']:
		# logging.info ('BATCH_' + BATCH_ID + ':faceset:'+ faceset_id +'delete succeed.')
		REMOVEABLE_FACESETS.remove(faceset_id)
		if len(REMOVEABLE_FACESETS) > 0:
			for removable_faceset in REMOVEABLE_FACESETS:
				delete_faceset(removable_faceset)
		return True

def process_faces_in_timeline(username):
	
	global MAX_FACESET, DELAY, FILE_THRESHOLD, FACE_THRESHOLD, \
		   FACE_THRESHOLD, SESSION_INQUEUE, SESSION_FACESET_MAP, FACE_POST_TIME_MAP, REMOVEABLE_FACESETS

	print ('current session in queue:' + str(len(SESSION_INQUEUE)))
	logging.info ('BATCH_' + BATCH_ID + ':process timeline for user: ' + username + '...')
	'''
		1. Detect faces in the 3 pictures and find out their positions and
			attributes
		2. Create persons using the face_id
		3. Create a new group and add those persons in it
		4. Train the model
		5. asyncronously wait for training to complete
	
		Args: 
			filename queue that represents a user's timeline
		Returns:
			groupping session_id
	'''
	# if session limit reach, wait until vancy appears
	while len(SESSION_INQUEUE) == MAX_FACESET:
		
		for session in SESSION_INQUEUE:
			
			try:
				rst = api.info.get_session(session_id=session)
			except:
				# session retrieve failed
				# detele corresponding faceset
				# delete session
				logging.info ('BATCH_' + BATCH_ID + ':retrieve session:'+ session +' failed')
				faceset_id = SESSION_FACESET_MAP[session][0]
				REMOVEABLE_FACESETS.append(faceset_id)
				
				if delete_faceset(faceset_id):
					logging.info ('BATCH_' + BATCH_ID + ':faceset:'+ faceset_id +' delete succeed')
					# INQUEUE -= 1
				SESSION_INQUEUE.remove(session)
				del SESSION_FACESET_MAP[session]
			
			if rst['status'] != 'INQUEUE':
				# either succeed or failed
				faceset_id = SESSION_FACESET_MAP[session][0]
				prev_username = SESSION_FACESET_MAP[session][1]
				REMOVEABLE_FACESETS.append(faceset_id)
				# delete used faceset before moving to next faceset
				if delete_faceset(faceset_id):
					logging.info ('BATCH_' + BATCH_ID + ':faceset:'+ faceset_id +' delete succeed')
					
				if session in SESSION_INQUEUE:
					SESSION_INQUEUE.remove(session)
					del SESSION_FACESET_MAP[session]
			
				# if succeed, process
				if rst['result']:
					logging.info ('BATCH_' + BATCH_ID + ':session('+ prev_username +'): ' + session + ' groupping result is ready for processing')
					USER = process_groupping_result(prev_username, rst['result'])
					if USER:
						logging.info ('BATCH_' + BATCH_ID + ':session('+ prev_username +'): ' + session + ' writing to file' ) 
						output.write(USER + '\n')
						logging.info ('BATCH_' + BATCH_ID + ':processing user: ' + prev_username + ' all completed.')
					else:
						logging.info ('BATCH_' + BATCH_ID + ':processing user: ' + prev_username + ' no groupped faces')
				else:
					logging.info ('BATCH_' + BATCH_ID + ':session('+ prev_username +'): ' + session + ' groupping failed')

		
		# if no session complete, before next iteration, sleep 5s
		if len(SESSION_INQUEUE) == MAX_FACESET:
			logging.info ('BATCH_' + BATCH_ID + ': faceset limit reached, sleep for ' + str(DELAY) + ' s...')
			time.sleep(DELAY)

	dir_prefix = '/public/lchi3/Pet/dog_data/' + username + '/pics'
	file_queue = listdir(dir_prefix)

	if len(file_queue) < FILE_THRESHOLD:
		logging.info ('BATCH_' + BATCH_ID + ':less than 30 pics, pass')
		return

	# 1. create a faceset for user timeline
	faceset_id = api.faceset.create(name=username)['faceset_id']
	# 2. detect faces to each image
	# collect all faces' ids across timeline
	face_id_str = ""
	total_file_counter = 0
	total_face_counter = 0
	file_counter = 0
	batch = 0

	start_time = time.time()

	for file in file_queue:
		
		total_file_counter += 1
		file_counter += 1
		
		logging.info ('BATCH_' + BATCH_ID + ':detecting faces for file. ' +  str(total_file_counter))
		
		try:
			faces = api.detection.detect(img=File(dir_prefix + '/' + file))['face']
			if len(faces) > 0:
				logging.info ('BATCH_' + BATCH_ID + ':detecting faces for file. ' +  str(total_file_counter) + ' succeed')
				# succeed, record it
				unixtime = file.split('_')[0]
				uid = file.split('_')[1]
				pid = file.split('_')[2]
				external_face_json = {'uid_pid':uid + '_' + pid, 'resp': faces, 'timestamp':unixtime}
				output_external_usage.write(json.dumps(external_face_json) + '\n')
			else:
				logging.info ('BATCH_' + BATCH_ID + ':detecting faces for file. ' +  str(total_file_counter) + ' no face found')
		except:
			logging.info ('BATCH_' + BATCH_ID + ':detecting faces for file. ' +  str(total_file_counter) + ' failed')
			continue

		total_face_counter += len(faces)
		for face in faces:
			face_id = face['face_id']
			# file name
			# unixtime-url.jpg
			timestamp = file.split('_')[0]
			FACE_POST_TIME_MAP[face_id] = timestamp 
			face_id_str += (face_id + ',')
		
		if file_counter == 10 and face_id_str == '':
			batch += 1
			logging.info ('BATCH_' + BATCH_ID + ':no face in the whole batch. ' +  str(batch) + ', pass')
			file_counter = 0
			
		if file_counter == 10 and face_id_str != '':
			batch += 1
			# trim the last comma
			face_id_str = face_id_str[:-1]
			# add faces to faceset
			logging.info ('BATCH_' + BATCH_ID + ':adding face batch. ' + str(batch))
			try:
				resp = api.faceset.add_face(face_id=face_id_str, faceset_id=faceset_id)
				if not resp['success']:
					logging.info ('BATCH_' + BATCH_ID + ':adding face batch. ' + str(batch) + 'failed:')
					logging.info(resp)
				else:
					logging.info ('BATCH_' + BATCH_ID + ':adding face batch. ' + str(batch) + 'succeed')
			except:
				logging.info ('BATCH_' + BATCH_ID + ':adding face batch. ' + str(batch) + 'failed:')
				logging.info (resp)
				face_id_str = ""
				file_counter = 0	
				continue
			
			face_id_str = ""
			file_counter = 0		
	
	if file_counter != 0 and face_id_str != '':
		face_id_str = face_id_str[:-1]
		try:
			resp = api.faceset.add_face(face_id=face_id_str, faceset_id=faceset_id)
			if not resp['success']:
				logging.info ('BATCH_' + BATCH_ID + ':adding face batch. ' + str(batch) + 'failed')
			else:
				logging.info ('BATCH_' + BATCH_ID + ':adding face batch. ' + str(batch) + 'succeed')
		except:
			logging.info ('BATCH_' + BATCH_ID + ':adding last batch of faces failed') 
			
		face_id_str = ""
		file_counter = 0

	logging.info ( 'detecting faces completed' )
	logging.info ( 'detecting elapsed time: ' + str(time.time() - start_time) + ' s' )

	if total_face_counter > FACE_THRESHOLD:
		# starts groupping asyncronously 
		logging.info ('BATCH_' + BATCH_ID + ':sending groupping resquest ...')
		try:
			resp = api.grouping.grouping(faceset_id=faceset_id)
			if resp['session_id']:
				session_id = resp['session_id']
				logging.info ('BATCH_' + BATCH_ID + ':groupping resquest for user:' + username +' sent successfully.')
				# INQUEUE += 1
				SESSION_INQUEUE.append(session_id)
				# when session complete, delete the corresponding faceset
				SESSION_FACESET_MAP[session_id] = [faceset_id, username]
				logging.info ('BATCH_' + BATCH_ID + ':session_id: ' + session_id)
				return session_id
			return 
		except:
			# remove the faceset
			REMOVEABLE_FACESETS.append(faceset_id)
			delete_faceset(faceset_id)
			logging.info ('BATCH_' + BATCH_ID + ':groupping resquest for user:' + username +' sent failed.')
			return
	else:
		# remove the faceset
		REMOVEABLE_FACESETS.append(faceset_id)
		delete_faceset(faceset_id)
		logging.info ('BATCH_' + BATCH_ID + ':too few faces to group, pass')
		return
 
# def process_timeline_captions(captions):
# 	'''
# 		process all captions of a user timeline to see
# 		if there is any repeating keywords
# 		Args: 
# 			all captions of a timeline
# 		Returns:  
# 			0 -> partner!
# 			1 -> child!
# 			2 -> child and partner!
# 	'''

def process_signle_person(faces, isUser):
	'''
		process person face_list
	'''
	face_ids = ""
	# check the face number
	# get faces by set
	face_count = 0
	face_list = []

	for face in faces:
		face_count += 1
		face_ids += ( face['face_id'] + ",")
		if face_count == 10:
			# trim lost comma
			face_ids = face_ids[:-1]
			# send request
			try:
				for face in api.info.get_face(face_id=face_ids)['face_info']:
					face_list.append(face)
			except:
				logging.info ('BATCH_' + BATCH_ID + ':get face failed')
			# clean
			face_ids = ""
			face_count = 0

	if face_ids != "":
		# trim lost comma
		face_ids = face_ids[:-1]
		try:
			for face in api.info.get_face(face_id=face_ids)['face_info']:
				face_list.append(face)
		except:
			logging.info ('BATCH_' + BATCH_ID + ':get face failed')

	if isUser:
		attribute = face_list[0]['attribute']
		smile = 0
		for face in face_list:
			smile += float(face['attribute']['smiling']['value'])
		smile = smile / len(face_list)

		return attribute, smile
	else:
		# times when this person are posted in user's timeline
		time_appeared = []
		for face in face_list:
			time_appeared.append(FACE_POST_TIME_MAP[face['face_id']])
		
		face_id = faces[0]['face_id']
		face_list = api.info.get_face(face_id=face_id)['face_info']
		attribute = face_list[0]['attribute']

		return attribute, time_appeared


def process_groupping_result(username, rst):
	logging.info ('BATCH_' + BATCH_ID + ':processing groupping result for user:' + username)

	if len(rst['group']) == 0:
		logging.info ('BATCH_' + BATCH_ID + ':no grouped faces for user:' + username)
		return None
	'''
		given the groupping result, get the user's attributes
		see if the user has a partner or not
		see if the user has child or not
		Args: 
			returned groupping result
		Returns: 
			{}
	'''
	# sort group list by group len
	groups = sorted(rst['group'], key=len)
	user = groups.pop()
	user_face_ids = ""
	## only consider groupped faces
	# USER
	attribute, smile = process_signle_person(user, True)
	USER = {'username': username, 'attribute': attribute, 'ave_smile': smile}
	# OTHERS
	USER['others'] = []
	for other in groups:
		attribute, time_appeared = process_signle_person(other, False)
		USER['others'].append({'attribute':attribute, 'times': time_appeared})

	# return json format
	return json.dumps(USER)


def process_tail_sessions():

	logging.info ('BATCH_' + BATCH_ID + ':process tail sessions')

	global MAX_FACESET, DELAY, FILE_THRESHOLD, FACE_THRESHOLD, \
	   FACE_THRESHOLD, SESSION_INQUEUE, SESSION_FACESET_MAP, FACE_POST_TIME_MAP, REMOVEABLE_FACESETS

	while len(SESSION_INQUEUE) > 0:
		temp_len = len(SESSION_INQUEUE)
		for session in SESSION_INQUEUE:
			rst = api.info.get_session(session_id=session)
			if rst['status'] != 'INQUEUE':
				
				# either succeed or failed
				faceset_id = SESSION_FACESET_MAP[session][0]
				REMOVEABLE_FACESETS.append(faceset_id)
				prev_username = SESSION_FACESET_MAP[session][1]
				
				# delete used faceset before moving to next faceset
				if session in SESSION_INQUEUE:
					SESSION_INQUEUE.remove(session)
					del SESSION_FACESET_MAP[session]
				
				delete_faceset(faceset_id)
				# INQUEUE -= 1
				
				# if succeed, process
				if rst['result']:
					logging.info ('BATCH_' + BATCH_ID + ':session('+ prev_username +'): ' + session + ' groupping result is ready for processing')
					USER = process_groupping_result(prev_username, rst['result'])
					if USER:
						logging.info ('BATCH_' + BATCH_ID + ':session('+ prev_username +'): ' + session + 'writing to file' ) 
						output.write(USER + '\n')
						logging.info ('BATCH_' + BATCH_ID + ':processing user: ' + prev_username + ' all completed.')
					else:
						logging.info ('BATCH_' + BATCH_ID + ':processing user: ' + prev_username + ' no groupped faces')
					# # USER = process_groupping_result(prev_username, rst['result'])
					# logging.info ('BATCH_' + BATCH_ID + ':session('+ prev_username +'): ' + session + 'writing to file' ) 
					# output.write(USER + '\n')
					# logging.info ('BATCH_' + BATCH_ID + ':processing user: ' + prev_username + ' all completed.' )
		
		# if no session complete, before next iteration, sleep 5s
		if len(SESSION_INQUEUE) == temp_len:
			logging.info ('BATCH_' + BATCH_ID + ':sleep for 5s...')
			time.sleep(DELAY)

# delete_faceset()
# start_time = time.time()
# # print(process_faces_in_timeline('_alessiadelorenzi_96'))
# # faceset_id = 'da209a060a2a4eb3a009acc2a11c307e'
# # print (api.grouping.grouping(faceset_id=faceset_id))
# print ( process_faces_in_timeline('_alessiadelorenzi_96') )
# print ( 'total elapsed time: ' + str(time.time() - start_time) + ' s' )