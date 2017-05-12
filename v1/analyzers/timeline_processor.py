# from pet_analyzer import analyze_pet
import face_analyzer 

from facepp import API 
from os import listdir
import time
import sys
from send_message import sendSMS

ARGS = sys.argv

# command line args
BATCH_ID = ARGS[1]
API_KEY = ARGS[2]
API_SECRET = ARGS[3]
# 0: facial analysis, 1: pet analysis
OPTION = ARGS[4]
# start index
startIdx = int(ARGS[5])
endIdx = startIdx + 800 # 8000 users per big-run, 10 threads
# dirs
data_dir = '/xxx/xxx/Pet/cat_data'
#data_dir = 'timeline' #/batch_' + BATCH_ID

err_dir = 'err/err_batch_' +  BATCH_ID + '.txt'
# input src
users = listdir(data_dir)
if '.DS_Store' in users:
	users.remove('.DS_Store')

users = users[startIdx: endIdx]
print ('processing' + str(len(users)) + ' users')
# if SKIP != 0:
# 	print ('skip ' + str(SKIP) + ' users')
# 	users = users[SKIP:]
# output src
# output = open(output_dir, 'w+')
err_output = open(err_dir, 'a+')
# initialize face analyzer
face_analyzer.init(BATCH_ID, API_KEY, API_SECRET)

user_count = 0

if int(OPTION) == 0:
	''' FACIAL ANALYSIS'''
	for username in users:

		user_count += 1
		print ('batch_' + BATCH_ID + ' is: working for user No.(' + str(user_count) +'/'+ str(len(users)) + \
			'): ' + username + '...')
		
		try:
			face_analyzer.process_faces_in_timeline(username)
		except:
			err_output.write('batch_' + BATCH_ID + ' is: process user:' + username + ' failed at' + str(time.ctime()))
		
		if user_count == len(users) / 4: # 25% checkpoint
			# 25%, send notification
			sendSMS('Batch ' + BATCH_ID + ' is 25 percent complete at ' + str(time.ctime()))
		elif user_count == len(users) / 2: # 50% checkpoint
			# half-way, send notification
			sendSMS('Batch ' + BATCH_ID + ' is 50 percent complete at ' + str(time.ctime()))
		elif user_count == 0.75 * len(users): # 75% checkpoint
			sendSMS('Batch ' + BATCH_ID + ' is 75 percent complete at ' + str(time.ctime()))

	# address the remaining in-progress sessions
	print ('processing tail sessions')
	try:
		face_analyzer.process_tail_sessions()
	except:
		print ('processing tail sessions ends with error')
		
	# service complete, send a message to corresponder
	print ('process tail sessions completed')
	sendSMS('Batch ' + BATCH_ID + ' is completed at ' + str(time.ctime()) + ', please have a check.')

elif int(OPTION) == 1:
	''' PET ANALYSIS'''
	pass 

