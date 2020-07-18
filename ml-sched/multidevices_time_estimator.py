import numpy as np
import json


def import_steptime_dataset():
    with open('multidevices_time_dataset.json') as json_file:
        json_data = json.load(json_file)

    conv_model_gpu_list = json_data['steptime_model_gpu']
    conv_model_cpu_list = json_data['steptime_model_cpu']

    return conv_model_gpu_list, conv_model_cpu_list


def generate_model_gpu_dataset(model_gpu_list):
    model_dataset = list()
    model_steptime = list()

    for model_info in model_gpu_list:
        model_info_dict = dict()

        data_cost_list = model_info['data_cost'].split('-')
        model_info_dict['input_size'] = int(data_cost_list[0])
        model_info_dict['channel_num'] = int(data_cost_list[1])
        model_info_dict['batch_size'] = int(data_cost_list[2])

        model_cost_list = model_info['model_cost'].split('-')
        model_info_dict['conv_layer_num'] = int(model_cost_list[0])
        model_info_dict['total_layer_num'] = int(model_cost_list[1])

        cpu_cost_list = model_info['cpu_cost'].split('-')
        model_info_dict['cpu_overall_input_size'] = int(cpu_cost_list[0])
        model_info_dict['cpu_overall_batch_size'] = int(cpu_cost_list[1])
        model_info_dict['cpu_overall_conv_layer'] = int(cpu_cost_list[2])
        model_info_dict['cpu_overall_total_layer'] = int(cpu_cost_list[3])

        environment_cost_list = model_info['environment_cost'].split('-')
        model_info_dict['flops'] = float(environment_cost_list[0])
        model_info_dict['platform'] = environment_cost_list[1]

        model_dataset.append(model_info_dict)

        model_step_time = model_info['step_time']
        model_steptime.append(model_step_time)

    return model_dataset, model_steptime


def generate_model_cpu_dataset(model_cpu_list):
    model_dataset = list()
    model_steptime = list()

    for model_info in model_cpu_list:
        model_info_dict = dict()

        data_cost_list = model_info['data_cost'].split('-')
        model_info_dict['input_size'] = int(data_cost_list[0])
        model_info_dict['channel_num'] = int(data_cost_list[1])
        model_info_dict['batch_size'] = int(data_cost_list[2])

        model_cost_list = model_info['model_cost'].split('-')
        model_info_dict['conv_layer_num'] = int(model_cost_list[0])
        model_info_dict['total_layer_num'] = int(model_cost_list[1])

        cpu_cost_list = model_info['cpu_cost'].split('-')
        model_info_dict['cpu_overall_input_size'] = int(cpu_cost_list[0])
        model_info_dict['cpu_overall_batch_size'] = int(cpu_cost_list[1])
        model_info_dict['cpu_overall_conv_layer'] = int(cpu_cost_list[2])
        model_info_dict['cpu_overall_total_layer'] = int(cpu_cost_list[3])

        environment_cost_list = model_info['environment_cost'].split('-')
        model_info_dict['threads_num'] = int(environment_cost_list[0])
        model_info_dict['IO'] = environment_cost_list[1]
        model_info_dict['platform'] = environment_cost_list[2]

        model_dataset.append(model_info_dict)

        model_step_time = model_info['step_time']
        model_steptime.append(model_step_time)

    return model_dataset, model_steptime


if __name__ == "__main__":
    model_gpu_list_json, model_cpu_list_json = import_steptime_dataset()
    gpu_job_info_list, gpu_job_steptime_list = generate_model_gpu_dataset(model_gpu_list_json)
    cpu_job_info_list, cpu_job_steptime_list = generate_model_cpu_dataset(model_cpu_list_json)
    print(gpu_job_info_list)
    print(gpu_job_steptime_list)
    print(cpu_job_info_list)
    print(cpu_job_steptime_list)