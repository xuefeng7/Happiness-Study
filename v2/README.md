# Happiness-Study
We leverage the state-of-the-art deep learning technologies and rich multimedia data from social networks to study human happiness involving multi-factors. This project has two versions, the first version focus on how pet effects our happiness, we analyzed user with different demographics (gender, race, having partner, having children) from both pet and none-pet owner groups. The second version is ongoing, and is  focusing on more and intersting factors, besides pet, that may impact our happiness.

## V1 Goals
V1 was successfully completed. The results from our data-driven approach were largely consistent with, and further confirmed,  our empirical commonsenses. More details can be found at [_The Effect of Pet on Happiness: A Large-scale Analysis using Social Multimedia_](https://www.xuefengpeng.com/publications/Pet.pdf)

## V2 Goals
Our goal is studying how having partner, children, pet, religion, and etc. impact our happiness and well-beings using social multimedia data.

## V1
There are three core files for v1, one is for retraining Google Inception-v3, and the other twos are pet analyzer, and face analyzer.

### How to retrain the Inception-V3 model
The code is from google, you may follow the instruction given at [_How to Retrain Inception's Final Layer for New_ Categories](https://www.tensorflow.org/tutorials/image_retraining). If Tensorflow with GPU or CPU version setup successfully, and acquired the [_retrain.py_](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py), you can just run:
```
python retrain.py \
--bottleneck_dir=tf_files/bottlenecks \
--how_many_training_steps 6000 \
--model_dir=tf_files/inception \
--output_graph=tf_files/retrained_graph.pb \
--output_labels=tf_files/retrained_labels.txt \
--image_dir tf_files/train_pet \
--random_crop=10 \
--random_scale=10 \
--random_brightness=10 \
--flip_left_right \
--train_batch_size=128 \
--testing_percentage=10 \
--validation_percentage=10
```
You can adjust the params as you see fit. Also, you may want to install tensorboard so as to visually view and evaluate your model performance.

### Face analyzer
We obtained around 20,000 Instagram users, and around 2-million posts from them. We need to recognize each timeline owner's face, detect his/her age, gender, and race. Besides, we also need to detect and recognize other frequenlty appeared faces, as those faces may belong to someone close to the timeline owner. i.e. partner, or child. 

#### _face_analyzer.py_ 
This file basically is calling Face++'s API serives including face detection and face grouping. We process all timeline posts user by user, and then derive a JSON string for each user.

```json
  {
	"username": "xxx",
	"attribute": {
		"gender": {
			"confidence": 99.957400000000007,
			"value": "Female"
		},
		"age": {
			"range": 5,
			"value": 7
		},
		"race": {
			"confidence": 84.206999999999994,
			"value": "Asian"
		},
		"smiling": {
			"value": 93.759699999999995
		}
	},
	"ave_smile": 83.540815454545466,
	"others": [{
		"attribute": {
			"gender": {
				"confidence": 99.973200000000006,
				"value": "Female"
			},
			"age": {
				"range": 7,
				"value": 27
			},
			"race": {
				"confidence": 82.450000000000003,
				"value": "Asian"
			},
			"smiling": {
				"value": 86.706699999999998
			}
		},
		"times": ["1428202998", "1488722913"]
	}, {
		"attribute": {
			"gender": {
				"confidence": 99.999399999999994,
				"value": "Female"
			},
			"age": {
				"range": 5,
				"value": 21
			},
			"race": {
				"confidence": 82.332700000000003,
				"value": "Asian"
			},
			"smiling": {
				"value": 72.154399999999995
			}
		},
		"times": ["1489716830", "1489716830"]
	}, {
		"attribute": {
			"gender": {
				"confidence": 90.432100000000005,
				"value": "Female"
			},
			"age": {
				"range": 8,
				"value": 13
			},
			"race": {
				"confidence": 59.357700000000001,
				"value": "White"
			},
			"smiling": {
				"value": 29.953299999999999
			}
		},
		"times": ["1488370317", "1488370317"]
	}, {
		"attribute": {
			"gender": {
				"confidence": 99.999899999999997,
				"value": "Female"
			},
			"age": {
				"range": 6,
				"value": 24
			},
			"race": {
				"confidence": 67.508499999999998,
				"value": "White"
			},
			"smiling": {
				"value": 42.716299999999997
			}
		},
		"times": ["1489778463", "1489716830", "1489716830", "1489716830"]
	}]
}
```
others is an array contains all faces that appears at user's timeline. The _times_ attributes indicate at when the face is post. Then, according to gender, age, and frequency of appearing, we can infer if current user has any partner or child. 

#### _timeline_processor.py_ 
This is a wrapper for face_analyzer, because there are 20,000 users, we processed them in "10 threads". We indexed each user, and each timeline_processor will take care several hundred of users. Note that to run in multiple threads, you may need to have multiple Face++ accounts. 

#### _send_message.py_
We need to know the progress of the processing, so for each "thread", when the 25%, 50%, 75% progresses are made, we send a text to ourself. This file plays this role. The messesge services we were using is: [nexmo](https://www.nexmo.com), it has very easy-to-use REST APIs. 

### Pet Analyzer
The pet analyzer processes all collected Instagram user's timeline posts, to see if any image contains pet (dog and cat). The analyzer is included in _pet_analyzer.py_. The analyzer produces three labels, they are dog, cat, and others (no any dog or cat). If you want to re-use it, just change the src data filename. Notice this source data file should contain JSON strings shown above (one line for a user). After processing, the final output would be like:

```json
{
	"username": "xxxx",
	"attribute": {
		"gender": {
			"confidence": 99.9994,
			"value": "Male"
		},
		"age": {
			"range": 5,
			"value": 25
		},
		"race": {
			"confidence": 38.9675,
			"value": "White"
		},
		"smiling": {
			"value": 5.02544
		}
	},
	"pet_labels": ["1471138295-others", "1482156655-others", "1472125973-others", "1480952857-others", "1469186407-dog", "1469015128-others", "1486231010-others", "1486819345-others", "1462096142-others", "1479063719-others", "1476388216-others", "1473789063-cat", "1477067371-others", "1472683653-others", "1468805723-others", "1469698262-others", "1469698583-others", "1478219063-others", "1474563146-others", "1470237089-others", "1472857098-others", "1480208393-others", "1489421735-cat", "1471545598-others", "1465860933-others", "1471013737-others", "1464534440-others", "1472553730-others", "1472313257-others", "1471512558-others", "1461703717-others", "1482868738-others", "1473022844-others", "1477066070-others", "1470748771-others", "1475158286-others", "1470170846-others", "1472662500-others", "1458394401-others", "1469294050-others", "1475614816-others", "1474978974-cat", "1476885220-others", "1475263464-cat", "1485876797-others", "1458505614-others", "1473174312-others", "1488211540-others", "1478558697-others", "1469792607-others", "1471348811-others", "1486655523-others", "1471713468-others", "1483996813-others", "1485019626-cat", "1472936145-cat", "1484443042-others", "1469761888-others", "1486292636-others", "1466714017-others", "1481287184-others", "1472052667-others", "1470404295-others", "1472053049-cat", "1474462269-others", "1469461581-others", "1465691223-others", "1486487160-cat", "1470179469-others", "1484318130-cat", "1458559480-others", "1486640042-others", "1458732163-cat", "1487966653-others", "1479744743-dog", "1458838095-others", "1458958873-others", "1473955289-others", "1468928950-others", "1469961580-others", "1487617591-others", "1477319345-others", "1468929397-others", "1475703644-others", "1466944774-others", "1476146319-others", "1476446842-others", "1472649693-others", "1480369214-others", "1460384161-others", "1489342933-others", "1474425557-cat", "1480926470-others", "1471709576-others", "1469971937-cat", "1474225059-others", "1475091823-others", "1475454750-cat", "1485366231-others", "1467650353-others"],
	"ave_smile": 69.37181,
	"others": [{
		"attribute": {
			"gender": {
				"confidence": 95.5325,
				"value": "Female"
			},
			"age": {
				"range": 7,
				"value": 39
			},
			"race": {
				"confidence": 49.9283,
				"value": "Asian"
			},
			"smiling": {
				"value": 86.4501
			}
		},
		"times": ["1469698262", "1469698262"]
	}, {
		"attribute": {
			"gender": {
				"confidence": 71.6286,
				"value": "Male"
			},
			"age": {
				"range": 5,
				"value": 28
			},
			"race": {
				"confidence": 60.877199999999995,
				"value": "Black"
			},
			"smiling": {
				"value": 30.7868
			}
		},
		"times": ["1475158286", "1487966653"]
	}, {
		"attribute": {
			"gender": {
				"confidence": 91.1122,
				"value": "Female"
			},
			"age": {
				"range": 5,
				"value": 13
			},
			"race": {
				"confidence": 69.7628,
				"value": "White"
			},
			"smiling": {
				"value": 19.6006
			}
		},
		"times": ["1470170846", "1470170846"]
	}, {
		"attribute": {
			"gender": {
				"confidence": 82.0674,
				"value": "Male"
			},
			"age": {
				"range": 8,
				"value": 28
			},
			"race": {
				"confidence": 98.21940000000001,
				"value": "White"
			},
			"smiling": {
				"value": 71.3487
			}
		},
		"times": ["1475614816", "1476446842"]
	}, {
		"attribute": {
			"gender": {
				"confidence": 67.1486,
				"value": "Male"
			},
			"age": {
				"range": 7,
				"value": 18
			},
			"race": {
				"confidence": 70.58290000000001,
				"value": "White"
			},
			"smiling": {
				"value": 3.48603
			}
		},
		"times": ["1480208393", "1470748771", "1472662500"]
	}]
}
```
The _pet_labels_ is an array of classified posts. "1489421735-cat" means the post, whose timestamp is 1489421735, contains one or more cats. By those labels, we can infer if a timeline owner is a pet owner or not.

