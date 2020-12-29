import tensorflow as tf
import argparse
from timeit import default_timer as timer

from relish.common.model_importer import ModelImporter
from relish.common.dataset_loader import load_dataset_para, load_train_dataset, load_eval_dataset


def profile_steptime(model_info_args):
    model_info = model_info_args.replace('leaky_relu', 'leakyrelu')
    hyperparameter_list = model_info.split('_')
    print(hyperparameter_list)
    job_id = hyperparameter_list[0]
    model_type = hyperparameter_list[1]
    model_layer = int(hyperparameter_list[2])
    batch_size = int(hyperparameter_list[3])
    model_optimizer = hyperparameter_list[4]
    learning_rate = float(hyperparameter_list[5])
    if hyperparameter_list[6] == 'leakyrelu':
        model_activation = 'leaky_relu'
    else:
        model_activation = hyperparameter_list[6]

    train_dataset = hyperparameter_list[7]

    img_w, img_h, num_chn, num_cls = load_dataset_para(train_dataset)
    train_feature_input, train_label_input = load_train_dataset(train_dataset)
    eval_feature_input, eval_label_input = load_eval_dataset(train_dataset)

    with tf.device(assign_device):
        feature_ph = tf.placeholder(tf.float32, [None, img_w, img_h, num_chn])
        label_ph = tf.placeholder(tf.int64, [None, num_cls])

        dm = ModelImporter(model_type, job_id, model_layer,
                           img_h, img_w, num_chn, num_cls,
                           batch_size, model_optimizer,
                           learning_rate, model_activation, False)
        model_entity = dm.get_model_entity()
        model_logit = model_entity.build(feature_ph, is_training=True)
        train_op = model_entity.train(model_logit, label_ph)

        #########################################################################
        # Traing the model
        #########################################################################

        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True

        step_time = 0
        step_count = 0
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            num_batch = train_label_input.shape[0] // batch_size
            for i in range(num_batch):
                start_time = timer()
                #print('step {} / {}'.format(i + 1, num_batch))
                batch_offset = i * batch_size
                batch_end = (i + 1) * batch_size

                train_feature_batch = train_feature_input[batch_offset:batch_end]
                train_label_batch = train_label_input[batch_offset:batch_end]

                sess.run(train_op, feed_dict={feature_ph: train_feature_batch,
                                              label_ph: train_label_batch})
                end_time = timer()
                dur_time = end_time - start_time
                #print("step time:", dur_time)
                step_time += dur_time
                step_count += 1

    avg_step_time = step_time / step_count * 1000

    return avg_step_time


if __name__ == '__main__':

    #########################################################################
    # Constant parameters
    #########################################################################

    rand_seed = 10000

    #########################################################################
    # Parameters read from command
    #########################################################################

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--result_path', required=True, action='store', help='the path of a result file')
    parser.add_argument('-d', '--running_device', required=True, action='store', help='identify the running device')

    args = parser.parse_args()
    result_path = args.result_path
    assign_device = args.running_device

    train_time_list = list()
    step_num_list = list()

    with open(result_path) as fp:
        line = fp.readline()
        while line:
            if line.startswith('**Job Result**: '):
                line_trim = line.replace('**Job Result**: ', '')
                model_info, result_info = line_trim.split(',')
                step = result_info.split('_')[1]
                # avg_time = profile_steptime(model_info)
                # train_time_list.append(avg_time)
                # print(train_time_list)
                step_num_list.append(step)
                # print(step_num_list)
                # total_time += avg_time * int(step)
            line = fp.readline()
    # print('total time: {}'.format(total_time))
    # print(train_time_list)
    # print('#########################')
    print(step_num_list)
