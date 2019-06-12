import sys


class DnnModel(object):
    def __init__(self, model_name, intance_name, model_layer, input_w, input_h, num_classes, batch_size, desired_accuracy):

        self.modelName = model_name
        self.instanceName = model_name + intance_name
        self.modelLayer = model_layer
        self.inputWidth = input_w
        self.inputHeight = input_h
        self.numClasses = num_classes
        self.batchSize = batch_size
        self.desiredAccuracy = desired_accuracy
        import_model = __import__(self.modelName)
        
        #the imported module and the regarding class name have to be the same
        clazz = getattr(import_model, self.modelName)
        self.modelEntity = clazz(self.instanceName, self.modelLayer, self.inputWidth, self.inputHeight, self.numClasses)
        

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
