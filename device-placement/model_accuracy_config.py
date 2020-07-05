import json

with open('model_accuracy_dataset.json') as json_file:
    json_data = json.load(json_file)

conv_model_list = json_data['conv_model']


