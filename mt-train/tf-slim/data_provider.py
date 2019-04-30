# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 2019
@author: Rui Liu
"""

import json
import os


def provide(annotation_path=None, images_dir=None):
    if not os.path.exists(annotation_path):
        raise ValueError('`annotation_path` does not exist.')

    annotation_json = open(annotation_path, 'r')
    annotation_list = json.load(annotation_json)
    image_files = []
    annotation_dict = {}
    for d in annotation_list:
        image_name = d.get('image_id')
        disease_class = d.get('class')
        if images_dir is not None:
            image_name = os.path.join(images_dir, image_name)
        image_files.append(image_name)
        annotation_dict[image_name] = disease_class
    return image_files, annotation_dict
