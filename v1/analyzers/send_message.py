import urllib
import urllib2

import time


def sendSMS(msg):
	params = {
	    'api_key': 'xxxx',
	    'api_secret': 'xxxx',
	    'to': 'xxxx',
	    'from': '12153833395',
	    'text': msg
	}

	url = 'https://rest.nexmo.com/sms/json?' + urllib.urlencode(params)

	request = urllib2.Request(url)
	request.add_header('Accept', 'application/json')
	response = urllib2.urlopen(request)
