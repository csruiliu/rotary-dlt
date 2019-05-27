import sys

class DnnModel(object):
    def __init__(self, model_name, model_layer, batch_size_range, desired_accuracy):
        self.modelName = model_name
        self.modelLayer = model_layer
        self.batchSizeRange = batch_size_range
        self.desiredAccuracy = desired_accuracy
        import_model = __import__(self.modelName)
        
        #the imported module and the regarding class name have to be the same
        clazz = getattr(import_model, self.modelName)
        self.modelEntity = clazz(self.modelName)

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



