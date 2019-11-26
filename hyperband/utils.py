import yaml

def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z 

if __name__ == "__main__":
    with open("config.yml", 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
    #data_path = cfg['data_path']
    
    hypermeter = cfg['hypermeter']
    batch_size = hypermeter['batch_size']
    opt_conf = hypermeter['optimizer']
    num_model_layer = hypermeter['num_model_layer']
    model = hypermeter['model'][0]
    print(model)
    
    
           
    