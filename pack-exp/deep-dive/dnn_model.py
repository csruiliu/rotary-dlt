import sys


class DnnModel(object):
    def __init__(self, model_type, model_instance_name, num_conv_layer, input_h, input_w, num_channels,
                 num_classes, batch_size, optimizer, learning_rate, activation):
        self.modelType = model_type
        self.modelInstanceName = model_instance_name
        self.numConvLayer = num_conv_layer
        self.inputWidth = input_w
        self.inputHeight = input_h
        self.numChannels = num_channels
        self.numClasses = num_classes
        self.batchSize = batch_size
        self.optimizer = optimizer
        self.learningRate = learning_rate
        self.activation = activation

        import_model = __import__(self.modelType)
        clazz = getattr(import_model, self.modelType)
        self.modelEntity = clazz(net_name=self.modelInstanceName, num_conv_layer=self.numConvLayer,
                                 input_h=self.inputHeight, input_w=self.inputWidth, num_channel=self.numChannels,
                                 num_classes=self.numClasses, batch_size=self.batchSize, opt=self.optimizer,
                                 learning_rate=self.learningRate, activation=self.activation)

    def getModelEntity(self):
        return self.modelEntity