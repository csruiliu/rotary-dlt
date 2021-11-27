import os
import matplotlib.pyplot as plt
import numpy as np

from rotary.estimator.rotary_estimator import RotaryEstimator
from rotary.estimator.relaqs_estimator import ReLAQSEstimator


def plot_accuracy_prediction(predict_acc_results,
                             predict_acc_results_baseline,
                             real_acc_results,
                             output_path):
    epoch_predict = len(predict_acc_results)
    epoch_predict_baseline = len(predict_acc_results_baseline)
    epoch_real = len(real_acc_results)

    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.gca()

    plt.plot(np.arange(1, epoch_predict + 1),
             predict_acc_results,
             linestyle='--',
             marker='^',
             label='Predict--Rotary')
    plt.plot(np.arange(1, epoch_predict_baseline + 1),
             predict_acc_results_baseline,
             linestyle='--',
             marker='p',
             label='Predict--ReLAQS')
    plt.plot(np.arange(1, epoch_real + 1),
             real_acc_results,
             linestyle='-',
             marker='.',
             markersize=6,
             label='Ground-truth')

    plt.xticks(range(1, epoch_real + 1), range(1, epoch_real + 1))
    plt.yticks(list(np.arange(0.4, 1.1, 0.2)), ['40%', '60%', '80%', '100%'])
    plt.xlabel("Training Epoch", fontsize=16)
    plt.ylabel("Training Accuracy", fontsize=16)

    ax.tick_params(axis='y', direction='in', labelsize=16)
    ax.tick_params(axis='x', direction='in', labelsize=13, pad=6)
    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    plt.legend(loc='upper left', fontsize=9)
    plt.savefig(output_path, format='png', bbox_inches='tight', pad_inches=0.05)


def main():
    #######################################################
    # sample jobs for prediction
    #######################################################

    test_bert_job = {
        'id': 1,
        'model': 'bert',
        'num_parameters': 14274305,
        'batch_size': 64,
        'training_data': 'stanford-lmrd',
        'opt': 'Adam',
        'learn_rate': 0.0001,
        'classes': 2
    }
    test_bert_job_acc = [
        0.8472399711608887,
        0.8930799961090088,
        0.9352399706840515,
        0.9728000164031982,
        0.9876800179481506
    ]

    test_lstm_job = {
        'id': 1,
        'model': 'bilstm',
        'num_parameters': 1614994,
        'batch_size': 64,
        'opt': 'Momentum',
        'learn_rate': 1e-05,
        'training_data': 'udtreebank',
        'classes': 17
    }
    test_lstm_job_acc = [
        0.0045885732397437096,
        0.004908268805593252,
        0.012118883430957794,
        0.03247463330626488,
        0.7305744290351868,
        0.8830961585044861,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414,
        0.8830881118774414
    ]

    test_cv_job = {
        'id': 1,
        'model': 'resnet',
        'num_parameters': 23513674,
        'batch_size': 32,
        'opt': 'SGD',
        'learn_rate': 0.1,
        'training_data': 'cifar10',
        'classes': 10
    }
    test_cv_job_acc = [
        0.42089999943971634,
        0.4938999991118908,
        0.5867000034451485,
        0.6345000052452088,
        0.6784000061452389,
        0.6882000038027763,
        0.7002000021934509,
        0.719200002849102,
        0.7315000009536743,
        0.7319000029563903,
        0.7286000055074692,
        0.7363000032305718,
        0.748000001013279,
        0.745300001502037,
        0.7477999997138977,
        0.7597999987006188,
        0.7435999992489815,
        0.7578999984264374,
        0.7533000007271766,
        0.7542000019550323
    ]

    #######################################################
    # init estimator
    #######################################################

    rotary_estimator = RotaryEstimator(topk=5, poly_deg=3)
    relaqs_estimator = ReLAQSEstimator()

    knowledgebase_path = '/home/ruiliu/Development/rotary/knowledgebase'

    # import all the accuracy files
    for f in os.listdir(knowledgebase_path):
        archive_file = knowledgebase_path + '/' + f
        rotary_estimator.import_knowledge_archive(archive_file)
        relaqs_estimator.import_knowledge_archive(archive_file)

    #######################################################
    # prediction
    #######################################################

    bert_epoch = 5
    lstm_epoch = 20
    cv_epoch = 20
    bert_acc_estimate = list()
    lstm_acc_estimate = list()
    cv_acc_estimate = list()
    cv_acc_estimate_baseline = list()

    bert_job_key = str(test_bert_job['id']) + '-' + test_bert_job['model']
    lstm_job_key = str(test_lstm_job['id']) + '-' + test_lstm_job['model']
    cv_job_key = str(test_cv_job['id']) + '-' + test_cv_job['model']

    '''
    for e in np.arange(0, bert_epoch):
        bert_acc_estimate.append(dle.predict(test_bert_job, input_x=e, mode='accuracy'))

    for e in np.arange(0, lstm_epoch):
        lstm_acc_estimate.append(dle.predict(test_lstm_job, input_x=e, mode='accuracy'))
    '''

    for e in np.arange(0, cv_epoch):
        cv_acc_estimate.append(rotary_estimator.predict(test_cv_job, input_x=e, mode='accuracy'))
        cv_acc_estimate_baseline.append(relaqs_estimator.predict(test_cv_job, input_x=e, mode='accuracy'))

    '''
    dle.import_knowledge_realtime(job_key=bert_job_key,
                                  accuracy=0.505079984664917,
                                  epoch=1)

    dle.import_knowledge_realtime(job_key=lstm_job_key,
                                  accuracy=0.7305744290351868,
                                  epoch=5)
    '''

    rotary_estimator.import_knowledge_realtime(job_key=cv_job_key,
                                               accuracy=0.5867000034451485,
                                               epoch=3)

    rotary_estimator.import_knowledge_realtime(job_key=cv_job_key,
                                               accuracy=0.6882000038027763,
                                               epoch=6)

    rotary_estimator.import_knowledge_realtime(job_key=cv_job_key,
                                               accuracy=0.7315000009536743,
                                               epoch=9)

    rotary_estimator.import_knowledge_realtime(job_key=cv_job_key,
                                               accuracy=0.7363000032305718,
                                               epoch=12)

    cv_acc_estimate = list()
    for e in np.arange(0, cv_epoch):
        cv_acc_estimate.append(rotary_estimator.predict(test_cv_job, input_x=e, mode='accuracy'))

    #######################################################
    # plot figure
    #######################################################

    out_path = '/home/ruiliu/Development/rotary/dl-estimator.png'
    plot_accuracy_prediction(cv_acc_estimate,
                             cv_acc_estimate_baseline,
                             test_cv_job_acc,
                             out_path)


if __name__ == "__main__":
    main()
