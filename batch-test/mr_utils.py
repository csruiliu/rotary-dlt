import sys
sys.path.append('/home/rui/Development/tf-exp/models/research/slim')
sys.path.append('/home/rui/Development/tf-exp/models/official')

import tensorflow as tf

def tensor_expand(tensor_Input,Num):
    tensor_Input = tf.expand_dims(tensor_Input,axis=0)
    tensor_Output = tensor_Input
    for i in range(Num-1):
        tensor_Output= tf.concat([tensor_Output,tensor_Input],axis=0)
    return tensor_Output

def get_one_hot_matrix(height,width,position):
    col_length = height
    row_length = width
    col_one_position = position[0]
    row_one_position = position[1]
    rows_num = height
    cols_num = width
 
    single_row_one_hot = tf.one_hot(row_one_position, row_length, dtype=tf.float32)
    single_col_one_hot = tf.one_hot(col_one_position, col_length, dtype=tf.float32)
 
    one_hot_rows = tensor_expand(single_row_one_hot, rows_num)
    one_hot_cols = tensor_expand(single_col_one_hot, cols_num)
    one_hot_cols = tf.transpose(one_hot_cols)
 
    one_hot_matrx = one_hot_rows * one_hot_cols
    return one_hot_matrx

def tensor_assign(tensor_input,position,value):
    shape = tensor_input.get_shape().as_list()
    height = shape[0]
    width = shape[1]
    h_index = position[0]
    w_index = position[1]
    one_hot_matrix = get_one_hot_matrix(height, width, position)
 
    new_tensor = tensor_input - tensor_input[h_index,w_index]*one_hot_matrix +one_hot_matrix*value
 
    return new_tensor

