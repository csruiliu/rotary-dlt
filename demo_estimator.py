import os
import matplotlib.pyplot as plt
import numpy as np

from estimator.dl_estimator import DLEstimator


def eval_accuracy_prediction():
    test_workload = [{
        "id": "1",
        "model": "mobilenet",
        "num_parameters": 2141234,
        "batch_size": 32,
        "opt": "SGD",
        "learn_rate": 0.01,
        "training_data": "cifar",
        "classes": 10
    }]

    test_job = {
        "id": "1",
        "model": "mobilenet",
        "num_parameters": 2141234,
        "batch_size": 32,
        "opt": "SGD",
        "learn_rate": 0.01,
        "training_data": "cifar",
        "classes": 10
    }

    real_accuracy = [
        0.4936999988555908,
        0.5672000032663346,
        0.613600004017353,
        0.6527000042796135,
        0.6697000038623809,
        0.6856000030040741,
        0.701600005030632,
        0.700500001758337,
        0.7007000043988227,
        0.6969000053405762,
        0.704100002348423,
        0.7104000025987625,
        0.7022000008821487,
        0.7142000058293343,
        0.711000003516674
    ]

    job_key = str(test_job['id']) + '-' + test_job['model']

    ###########################################
    # draw 4 subfigures
    ###########################################

    fig, axs = plt.subplots(2, 2, figsize=(28, 17))

    ###########################################
    # original figure
    ###########################################

    dle.prepare_workload(test_workload)

    acc_estimate, coefs = dle.predict_accuracy(test_job, 1)

    predict_dict = dle.get_predict_dict()[job_key]
    accuracy_data = predict_dict['accuracy']
    epoch_data = [x + 1 for x in predict_dict['epoch']]

    for i in range(1, topk + 1):
        for j in range(0, 5):
            epoch_data.pop(15 * i)
            accuracy_data.pop(15 * i)

    axs[0, 0].spines["top"].set_linewidth(5)
    axs[0, 0].spines["left"].set_linewidth(5)
    axs[0, 0].spines["right"].set_linewidth(5)
    axs[0, 0].spines["bottom"].set_linewidth(5)

    axs[0, 0].plot(epoch_data,
                   accuracy_data,
                   color='cornflowerblue',
                   marker='o',
                   markersize=10,
                   linestyle='None')

    axs[0, 0].plot(np.arange(1, 16),
                   real_accuracy,
                   color="green",
                   linestyle='-',
                   linewidth=5,
                   label='real')

    axs[0, 0].plot(np.arange(1, 16),
                   np.polyval(coefs, np.arange(1, 16)),
                   color="red",
                   linestyle='--',
                   linewidth=5,
                   label='estimate')

    axs[0, 0].set_xlabel('Epochs\n(a)', fontsize=44, labelpad=12)
    axs[0, 0].set_ylabel('Accuracy', fontsize=44)
    axs[0, 0].set_yticks(np.arange(0, 1.01, 0.2))
    axs[0, 0].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs[0, 0].set_xticks(np.arange(1, 15.1, 1))

    axs[0, 0].grid(True, linewidth=3, linestyle='--')

    axs[0, 0].tick_params(axis='y', direction='in', labelsize=36)
    axs[0, 0].tick_params(axis='x', direction='in', bottom=False, labelsize=38, pad=8)

    ###########################################
    # add 2 actual points
    ###########################################

    dle.prepare_workload(test_workload)

    dle.add_actual_data(job_key, 0.4936999988555908, 0)
    dle.add_actual_data(job_key, 0.613600004017353, 2)

    acc_estimate, coefs = dle.predict_accuracy(test_job, 2)

    predict_dict = dle.get_predict_dict()[job_key]
    accuracy_data = predict_dict['accuracy']
    epoch_data = [x + 1 for x in predict_dict['epoch']]

    for i in range(1, topk + 1):
        for j in range(0, 5):
            epoch_data.pop(15 * i)
            accuracy_data.pop(15 * i)

    axs[0, 1].spines["top"].set_linewidth(5)
    axs[0, 1].spines["left"].set_linewidth(5)
    axs[0, 1].spines["right"].set_linewidth(5)
    axs[0, 1].spines["bottom"].set_linewidth(5)

    axs[0, 1].plot(epoch_data,
                   accuracy_data,
                   color='cornflowerblue',
                   marker='o',
                   markersize=10,
                   linestyle='None')

    axs[0, 1].plot(np.arange(1, 16),
                   real_accuracy,
                   color="green",
                   linestyle='-',
                   linewidth=5,
                   label='real')

    axs[0, 1].plot(np.arange(1, 16),
                   np.polyval(coefs, np.arange(1, 16)),
                   color="red",
                   linestyle='--',
                   linewidth=5,
                   label='estimate')

    # read data
    axs[0, 1].plot([1], [0.4936999988555908],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle=None,
                   label='real_one')

    axs[0, 1].plot([3], [0.613600004017353],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle=None,
                   label='real_one')

    axs[0, 1].set_xlabel('Epochs\n(b)', fontsize=44, labelpad=12)
    axs[0, 1].set_ylabel('Accuracy', fontsize=44)
    axs[0, 1].set_yticks(np.arange(0, 1.01, 0.2))
    axs[0, 1].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs[0, 1].set_xticks(np.arange(1, 15.1, 1))

    axs[0, 1].tick_params(axis='y', direction='in', labelsize=36)
    axs[0, 1].tick_params(axis='x', direction='in', bottom=False, labelsize=38, pad=8)

    axs[0, 1].grid(True, linewidth=2, linestyle='--')

    ###########################################
    # add 4 actual points
    ###########################################

    dle.prepare_workload(test_workload)

    dle.add_actual_data(job_key, 0.4936999988555908, 0)

    dle.add_actual_data(job_key, 0.613600004017353, 2)

    dle.add_actual_data(job_key, 0.6697000038623809, 4)

    dle.add_actual_data(job_key, 0.701600005030632, 6)

    acc_estimate, coefs = dle.predict_accuracy(test_job, 8)

    predict_dict = dle.get_predict_dict()[job_key]
    accuracy_data = predict_dict['accuracy']
    epoch_data = [x + 1 for x in predict_dict['epoch']]

    for i in range(1, topk + 1):
        for j in range(0, 5):
            epoch_data.pop(15 * i)
            accuracy_data.pop(15 * i)

    axs[1, 0].spines["top"].set_linewidth(5)
    axs[1, 0].spines["left"].set_linewidth(5)
    axs[1, 0].spines["right"].set_linewidth(5)
    axs[1, 0].spines["bottom"].set_linewidth(5)

    axs[1, 0].plot(np.arange(1, 16),
                   real_accuracy,
                   color="green",
                   linewidth=5,
                   linestyle='-',
                   label='Ground-truth Acc')

    axs[1, 0].plot(np.arange(1, 16),
                   np.polyval(coefs, np.arange(1, 16)),
                   color="red",
                   linewidth=5,
                   linestyle='--',
                   label='Estimate Acc')

    axs[1, 0].plot(epoch_data,
                   accuracy_data,
                   color='cornflowerblue',
                   marker='o',
                   markersize=10,
                   linestyle='None',
                   label='Archived Job Acc')

    # real data
    axs[1, 0].plot([1], [0.4936999988555908],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None',
                   label='Active Job Acc')

    axs[1, 0].plot([3], [0.613600004017353],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None',
                   label='Active Job Acc')

    axs[1, 0].plot([5], [0.6697000038623809],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None',
                   label='Active Job Acc')

    axs[1, 0].plot([7], [0.701600005030632],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None',
                   label='Active Job Acc')

    axs[1, 0].set_xlabel('Epochs\n(c)', fontsize=44, labelpad=12)
    axs[1, 0].set_ylabel('Accuracy', fontsize=44)
    axs[1, 0].set_yticks(np.arange(0, 1.01, 0.2))
    axs[1, 0].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs[1, 0].set_xticks(np.arange(1, 15.1, 1))

    axs[1, 0].tick_params(axis='y', direction='in', labelsize=36)
    axs[1, 0].tick_params(axis='x', direction='in', bottom=False, labelsize=38, pad=8)

    axs[1, 0].grid(True, linewidth=3, linestyle='--')

    ###########################################
    # add 6 actual points
    ###########################################

    dle.prepare_workload(test_workload)

    dle.add_actual_data(job_key, 0.4936999988555908, 0)

    dle.add_actual_data(job_key, 0.613600004017353, 2)

    dle.add_actual_data(job_key, 0.6697000038623809, 4)

    dle.add_actual_data(job_key, 0.701600005030632, 6)

    dle.add_actual_data(job_key, 0.7007000043988227, 8)

    dle.add_actual_data(job_key, 0.704100002348423, 10)

    acc_estimate, coefs = dle.predict_accuracy(test_job, 13)

    predict_dict = dle.get_predict_dict()[job_key]
    accuracy_data = predict_dict['accuracy']
    epoch_data = [x + 1 for x in predict_dict['epoch']]

    for i in range(1, topk + 1):
        for j in range(0, 5):
            epoch_data.pop(15 * i)
            accuracy_data.pop(15 * i)

    axs[1, 1].spines["top"].set_linewidth(5)
    axs[1, 1].spines["left"].set_linewidth(5)
    axs[1, 1].spines["right"].set_linewidth(5)
    axs[1, 1].spines["bottom"].set_linewidth(5)

    axs[1, 1].plot(np.arange(1, 16),
                   real_accuracy,
                   color="green",
                   linewidth=5,
                   linestyle='-',
                   label='Ground-truth Acc')

    axs[1, 1].plot(np.arange(1, 16),
                   np.polyval(coefs, np.arange(1, 16)),
                   color="red",
                   linewidth=5,
                   linestyle='--',
                   label='Estimate Acc')

    axs[1, 1].plot(epoch_data,
                   accuracy_data,
                   color='cornflowerblue',
                   marker='o',
                   markersize=10,
                   linestyle='None',
                   label='Archived Job Acc')

    # real data
    axs[1, 1].plot([1], [0.4936999988555908],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None',
                   label='Active Job Acc')

    axs[1, 1].plot([3], [0.613600004017353],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None')

    axs[1, 1].plot([5], [0.6697000038623809],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None')

    axs[1, 1].plot([7], [0.701600005030632],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None')

    axs[1, 1].plot([9], [0.7007000043988227],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None')

    axs[1, 1].plot([11], [0.704100002348423],
                   color="blueviolet",
                   marker='D',
                   markersize=18,
                   linestyle='None')

    axs[1, 1].set_xlabel('Epochs\n(d)', fontsize=44, labelpad=12)
    axs[1, 1].set_ylabel('Accuracy', fontsize=44)
    axs[1, 1].set_yticks(np.arange(0, 1.01, 0.2))
    axs[1, 1].set_yticklabels(['0%', '20%', '40%', '60%', '80%', '100%'])
    axs[1, 1].set_xticks(np.arange(1, 15.1, 1))

    axs[1, 1].tick_params(axis='y', direction='in', labelsize=36)
    axs[1, 1].tick_params(axis='x', direction='in', bottom=False, labelsize=38, pad=8)

    axs[1, 1].grid(True, linewidth=3, linestyle='--')

    plt.tight_layout()

    plt.legend(loc='upper center', bbox_to_anchor=(-0.16, 2.7), ncol=4, fontsize=38)

    plt.savefig(outpath, format='pdf', bbox_inches='tight', pad_inches=0.05)


def eval_epoch_prediction():
    test_workload = [{
        "id": "1",
        "model": "mobilenet",
        "num_parameters": 2141234,
        "batch_size": 32,
        "opt": "SGD",
        "learn_rate": 0.01,
        "training_data": "cifar",
        "classes": 10
    }]

    test_job = {
        "id": "1",
        "model": "mobilenet",
        "num_parameters": 2141234,
        "batch_size": 32,
        "opt": "SGD",
        "learn_rate": 0.01,
        "training_data": "cifar",
        "classes": 10
    }

    real_accuracy = [
        0.4936999988555908,
        0.5672000032663346,
        0.613600004017353,
        0.6527000042796135,
        0.6697000038623809,
        0.6856000030040741,
        0.701600005030632,
        0.700500001758337,
        0.7007000043988227,
        0.6969000053405762,
        0.704100002348423,
        0.7104000025987625,
        0.7022000008821487,
        0.7142000058293343,
        0.711000003516674
    ]

    job_key = str(test_job['id']) + '-' + test_job['model']

    dle.prepare_workload(test_workload)

    epoch_estimate, coefs = dle.predict_epoch(test_job, 0.613600004017353)

    print(epoch_estimate)


if __name__ == "__main__":
    outpath = '/home/ruiliu/Development/dl-estimator.pdf'

    topk = 10

    dle = DLEstimator(topk)

    # read all the accuracy file
    for f in os.listdir('./knowledgebase'):
        model_acc_file = os.getcwd() + '/knowledgebase/' + f
        dle.import_accuracy_dataset(model_acc_file)

    eval_epoch_prediction()
