import sys


class DnnModel(object):
    def __init__(self, model_name, intance_name, model_layer, input_w, input_h, num_classes, batch_size, optimizer):

        self.modelName = model_name
        self.instanceName = model_name + intance_name
        self.modelLayer = model_layer
        self.inputWidth = input_w
        self.inputHeight = input_h
        self.numClasses = num_classes
        self.batchSize = batch_size
        self.optimizer = optimizer
        import_model = __import__(self.modelName)
        
        #the imported module and the regarding class name have to be the same
        clazz = getattr(import_model, self.modelName)
        self.modelEntity = clazz(net_name=self.instanceName, model_layer=self.modelLayer, 
                                 input_h=self.inputHeight, input_w=self.inputWidth, batch_size=self.batchSize, 
                                 num_classes=self.numClasses, opt=self.optimizer)
        
    def getModelLayer(self):
        return self.modelLayer

    def getInstanceName(self):
        return self.instanceName

    def getModelName(self):
        return self.modelName

    def getModelEntity(self):
        return self.modelEntity

    def getBatchSize(self):
        return self.batchSize

    def getDesiredAccuracy(self):
        return self.desiredAccuracy
