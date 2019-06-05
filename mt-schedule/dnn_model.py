import sys


class DnnModel(object):
    def __init__(self, model_name, intance_name, model_layer, input_w, input_h, num_classes, batch_size_range, desired_accuracy):

        self.modelName = model_name
        self.modelLayer = model_layer
        self.inputWidth = input_w
        self.inputHeight = input_h
        self.num_classes = num_classes
        self.batchSizeRange = batch_size_range
        self.desiredAccuracy = desired_accuracy
        import_model = __import__(self.modelName)
        
        #the imported module and the regarding class name have to be the same
        clazz = getattr(import_model, self.modelName)
        self.modelEntity = clazz(self.modelName + intance_name, self.modelLayer, self.inputWidth, self.inputHeight, self.num_classes)
        

    def getModelLayer(self):
        return self.modelLayer

    def getModelName(self):
        return self.modelName

    def getModelEntity(self):
        return self.modelEntity

    def getBatchSizeRange(self):
        return self.batchSizeRange

    def getDesiredAccuracy(self):
        return self.desiredAccuracy
