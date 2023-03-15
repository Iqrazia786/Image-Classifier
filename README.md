# Image-Classifier
Project code for Udacity's AI Programming with Python Nanodegree program. In this project, code developed for an image classifier built with PyTorch, then converted into a command line applications: train.py, predict.py.

The image classifier to recognize different species of flowers. Dataset contains 102 flower categories.

# Command line applications train.py and predict.py
For command line applications there is an option to select either densenet121 or VGG13 models.

Following argumentsare mandatory or optional for train.py

'data_dir'. 'Provide data directory. Mandatory argument', type = str
'--save_dir'. 'Provide saving directory. Optional argument', type = str
'--arch'. 'Vgg13 can be used if this argument specified, otherwise densenet121 will be used', type = str
'--lrn'. 'Learning rate, default value 0.001', type = float
'--hidden_unit1', type=int, default=1024, help='hidden_unit1')
'--hidden_unit4', type=int, default=102, help='hidden_unit4')
'--epochs'. 'Number of epochs', type = int
'--GPU'. "Option to use GPU", type = str
Following arguments are mandatory or optional for predict.py

'image_dir'. 'Provide path to image. Mandatory argument', type = str
'load_dir'. 'Provide path to checkpoint. Mandatory argument', type = str
'--top_k'. 'Top K most likely classes. Optional', type = int
'--category_names'. 'Mapping of categories to real names. JSON file name to be provided. Optional', type = str
'--GPU'. "Option to use GPU. Optional", type = str
