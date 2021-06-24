import os
import matplotlib.pyplot as plt
import numpy as np

from estimator.dl_estimator import DLEstimator


def eval_epoch_prediction():
    test_workload = [{
        "id": "1",
        "model": "resnet",
        "num_parameters": 11173962,
        "batch_size": 32,
        "opt": "SGD",
        "learn_rate": 0.1,
        "training_data": "cifar10",
        "classes": 10
    }]

    test_job = {
        "id": "1",
        "model": "resnet",
        "num_parameters": 11173962,
        "batch_size": 32,
        "opt": "SGD",
        "learn_rate": 0.1,
        "training_data": "cifar10",
        "classes": 10
    }

    job_key = str(test_job['id']) + '-' + test_job['model']

    dle.prepare_workload(test_workload)

    epoch_estimate_zero = list()
    for acc in real_accuracy:
        epoch_estimate = dle.predict_epoch(test_job, acc)
        epoch_estimate_zero.append(epoch_estimate)
    epoch_estimate_zero = [round(item) for item in epoch_estimate_zero]
    epoch_estimate_zero = [0 if i < 0 else i for i in epoch_estimate_zero]

    ####################################

    dle.add_actual_data(job_key=job_key,
                        accuracy=0.4936999988555908,
                        epoch=1)
    epoch_estimate_one = list()
    for acc in real_accuracy:
        epoch_estimate = dle.predict_epoch(test_job, acc)
        epoch_estimate_one.append(epoch_estimate)
    epoch_estimate_one = [round(item) for item in epoch_estimate_one]
    epoch_estimate_one = [0 if i < 0 else i for i in epoch_estimate_one]

    ####################################

    dle.add_actual_data(job_key=job_key,
                        accuracy=0.613600004017353,
                        epoch=3)
    epoch_estimate_three = list()
    for acc in real_accuracy:
        epoch_estimate = dle.predict_epoch(test_job, acc)
        epoch_estimate_three.append(epoch_estimate)
    epoch_estimate_three = [round(item) for item in epoch_estimate_three]
    epoch_estimate_three = [0 if i < 0 else i for i in epoch_estimate_three]

    ####################################

    dle.add_actual_data(job_key=job_key,
                        accuracy=0.6697000038623809,
                        epoch=5)
    epoch_estimate_five = list()
    for acc in real_accuracy:
        epoch_estimate = dle.predict_epoch(test_job, acc)
        epoch_estimate_five.append(epoch_estimate)
    epoch_estimate_five = [round(item) for item in epoch_estimate_five]
    epoch_estimate_five = [0 if i < 0 else i for i in epoch_estimate_five]

    ####################################

    dle.add_actual_data(job_key=job_key,
                        accuracy=0.701600005030632,
                        epoch=7)
    epoch_estimate_seven = list()
    for acc in real_accuracy:
        epoch_estimate = dle.predict_epoch(test_job, acc)
        epoch_estimate_seven.append(epoch_estimate)
    epoch_estimate_seven = [round(item) for item in epoch_estimate_seven]
    epoch_estimate_seven = [0 if i < 0 else i for i in epoch_estimate_seven]

    return epoch_estimate_zero, epoch_estimate_one, epoch_estimate_three, epoch_estimate_five, epoch_estimate_seven


def plot_figure(epoch_estimate_zero,
                epoch_estimate_one,
                epoch_estimate_three,
                epoch_estimate_five,
                epoch_estimate_seven):
    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.gca()

    plt.plot(np.arange(0, 11), np.arange(0, 11), linestyle='--', label='Ground-truth')
    plt.plot(np.arange(0, 11),
             [0, 0, 0, 1, 1, 2, 2] + epoch_estimate_zero[6:10],
             marker='o',
             markersize=4,
             label='Estimate with archived data')
    plt.plot(np.arange(0, 11),
             [0] + epoch_estimate_one[0:10],
             marker='^',
             markersize=4,
             label='Estimate with archived & 1st-epoch data')
    plt.plot(np.arange(0, 11),
             [0] + epoch_estimate_three[0:10],
             marker='*',
             markersize=6,
             label='Estimate with archived & 3rd-epoch data')
    plt.plot(np.arange(0, 11),
             [0] + epoch_estimate_five[0:10],
             marker='D',
             markersize=4,
             label='Estimate with archived & 5th-epoch data')
    plt.plot(np.arange(0, 11),
             [0] + epoch_estimate_seven[0:10],
             marker='s',
             markersize=4,
             label='Estimate with archived & 7th-epoch data')

    plt.xticks(range(0, 11), ['0%', '49%', '57%', '61%', '65%', '67%', '69%', '70%', '70%', '70%', '70%'])
    plt.yticks(range(0, 11), range(0, 11))

    plt.ylabel("Training Epoch", fontsize=16)
    plt.xlabel("Training Accuracy", fontsize=16)

    ax.tick_params(axis='y', direction='in', labelsize=16)
    ax.tick_params(axis='x', direction='in', labelsize=13, pad=6)
    ax.grid(which='major', axis='x', linestyle='--')
    ax.spines["top"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)

    plt.legend(loc='upper left', fontsize=9)
    # plt.grid(True, linewidth=1, linestyle='--')
    plt.savefig(outpath, format='pdf', bbox_inches='tight', pad_inches=0.05)


if __name__ == "__main__":
    outpath = '/home/ruiliu/Development/dl-estimator.pdf'

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

    dle = DLEstimator(topk=10, poly_deg=3)

    # read all the accuracy file
    for f in os.listdir('./knowledgebase'):
        model_acc_file = os.getcwd() + '/knowledgebase/' + f
        dle.import_accuracy_dataset(model_acc_file)

    (epoch_estimate_zero,
     epoch_estimate_one,
     epoch_estimate_three,
     epoch_estimate_five,
     epoch_estimate_seven) = eval_epoch_prediction()

    plot_figure(epoch_estimate_zero,
                epoch_estimate_one,
                epoch_estimate_three,
                epoch_estimate_five,
                epoch_estimate_seven)

'''
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
'''