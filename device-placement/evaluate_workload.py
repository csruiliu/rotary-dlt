import tensorflow as tf
import numpy as np
from operator import itemgetter
from multiprocessing import Process, Pipe

from dnn_model import DnnModel
from img_utils import *
import config as cfg_yml


def generate_workload():
    workload_list = list()

    np.random.seed(randSeed)
    model_name_abbr = np.random.choice(randSeed, workloadNum, replace=False).tolist()

    for idx, mt in enumerate(workloadModelType):
        workload_batchsize_index_list = np.random.choice(len(workloadBatchSize), workloadModelNum[idx], replace=False).tolist()
        workload_batchsize_list = list(itemgetter(*workload_batchsize_index_list)(workloadBatchSize))

        for bs in workload_batchsize_list:
            workload_list.append([mt, bs, model_name_abbr.pop()])

    return workload_list


def evaluate_model():
    acc_list = list()
    sum_acc = 0
    for job in workload_placement:
        parent_conn, child_conn = Pipe()
        p = Process(target=evaluate_single_job, args=(job[0], job[1], job[2], child_conn))
        p.start()
        single_acc = parent_conn.recv()
        acc_list.append(job[0]+str(job[1])+':'+single_acc)
        sum_acc += single_acc
        parent_conn.close()
        p.join()

    print('Accuracy List:', acc_list)
    print('Max', max(acc_list))
    print('75Q:', np.quantile(acc_list, 0.75))
    print('Median:', np.quantile(acc_list, 0.5))
    print('25Q:', np.quantile(acc_list, 0.25))
    print('Min', min(acc_list))
    return sum_acc / workloadNum


def evaluate_single_job(model_type, batch_size, model_instance, conn):
    features = tf.placeholder(tf.float32, [None, imgWidth, imgHeight, numChannels])
    labels = tf.placeholder(tf.int64, [None, numClasses])

    dm = DnnModel(model_type, str(model_instance), 1, imgHeight, imgWidth, numChannels, numClasses, batch_size, 'Adam',
                  0.0001, 'relu', False)
    modelEntity = dm.getModelEntity()
    modelLogit = modelEntity.build(features)
    evalOps = modelEntity.evaluate(modelLogit, labels)

    saver = tf.train.Saver()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    #image_list = sorted(os.listdir(image_path_raw))
    model_ckpt_path = ckpt_path + '/' + model_type + '_' + str(batch_size)

    with tf.Session(config=config) as sess:
        saver.restore(sess, model_ckpt_path + '/' + model_type + '_' + str(batch_size))
        X_data_eval = load_imagenet_bin(test_image_path_bin, numChannels, imgWidth, imgHeight)
        Y_data_eval = load_imagenet_labels_onehot(test_label_path, numClasses)
        acc_arg = evalOps.eval({features: X_data_eval, labels: Y_data_eval})
    conn.send(acc_arg)
    conn.close()
    print("Accuracy:", acc_arg)


if __name__ == '__main__':

    #########################
    # Parameters
    #########################

    imgWidth = cfg_yml.img_width
    imgHeight = cfg_yml.img_height
    numChannels = cfg_yml.num_channels
    numClasses = cfg_yml.num_classes
    randSeed = cfg_yml.rand_seed

    workloadModelType = cfg_yml.workload_model_type
    workloadModelNum = cfg_yml.workload_model_num
    workloadBatchSize = cfg_yml.workload_batch_size
    workloadActivation = cfg_yml.workload_activation
    workloadOptimizer = cfg_yml.workload_opt
    workloadLearnRate = cfg_yml.workload_learning_rate
    workloadNumLayer = cfg_yml.workload_num_layer

    #########################
    # Build Workload
    #########################

    workloadNum = sum(workloadModelNum)
    workload_placement = generate_workload()

    #########################
    # Model Placement
    #########################

    image_path_raw = cfg_yml.imagenet_t10k_img_path
    image_path_bin = cfg_yml.imagenet_t10k_bin_path
    label_path = cfg_yml.imagenet_t10k_label_path
    ckpt_path = cfg_yml.ckpt_path

    test_image_path_raw = cfg_yml.imagenet_t1k_img_path
    test_image_path_bin = cfg_yml.imagenet_t1k_bin_path
    test_label_path = cfg_yml.imagenet_t1k_label_path

    avg_acc = evaluate_model()
    print('Average Accuracy:', avg_acc)