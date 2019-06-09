import tensorflow as tf
from timeit import default_timer as timer
from multiprocessing import Pool

class Schedule(object):
    def __init__(self, model_collection, input_w, input_h, num_classes):
        self.modelCollection = model_collection
        self.modelEntityCollection = []
        self.logitCollection = []
        self.crossEntropyCollection = []
        self.trainStepColllection = []
        self.scheduleCollection = []

        self.features = tf.placeholder(tf.float32, [None, input_w, input_h, 3])
        self.labels = tf.placeholder(tf.int64, [None, num_classes])


    def buildModels(self):
        for midx in self.modelCollection:
            modelEntity = midx.getModelEntity()
            self.modelEntityCollection.append(modelEntity) 
            modelLogit = modelEntity.build(self.features)
            self.logitCollection.append(modelLogit)

        for lidx in self.modelEntityCollection:
            print(lidx.getModelInstanceName())
            print(lidx.getModelMemSize())
    

    def scheduleNaive(self):
        for lit in self.logitCollection:
            #print(lit)
            scheduleUnit = []
            scheduleUnit.append(lit)
            self.scheduleCollection.append(lit)

    def executeSch(self, X_train, Y_train):
        total_time = 0
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            mini_batches = 10
            num_batch = Y_train.shape[0] // mini_batches            
            for schUntit in self.scheduleCollection:
                for i in range(num_batch):
                    print('step %d / %d' %(i+1, num_batch))
                    X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                    Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                    start_time = timer()
                    sess.run(schUntit, feed_dict={self.features: X_mini_batch_feed, self.labels: Y_mini_batch_feed})
                    end_time = timer()
                    total_time += end_time - start_time
                
                print("training time for 1 epoch:", total_time)  

    def execute

    def testSingleModel(self, model, X_train, Y_train):
        modelEntity = model.getModelEntity()
        modelLogit = modelEntity.build(self.features)
        modelCrossEntropy = modelEntity.cost(modelLogit, self.labels)

        with tf.name_scope('optimizer_test'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                modelTrainStep = tf.train.AdamOptimizer(1e-4).minimize(modelCrossEntropy)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            mini_batches = 10
            num_batch = Y_train.shape[0] // mini_batches

            total_time = 0

            for i in range(num_batch):
                print('step %d / %d' %(i+1, num_batch))
                X_mini_batch_feed = X_train[num_batch:num_batch + mini_batches,:,:,:]
                Y_mini_batch_feed = Y_train[num_batch:num_batch + mini_batches,:]
                start_time = timer()

                sess.run(modelTrainStep, feed_dict={self.features: X_mini_batch_feed, self.labels: Y_mini_batch_feed})
                end_time = timer()
                total_time += end_time - start_time
            print("training time for 1 epoch:", total_time)


    def showAllModelInstances(self):
        for idx in self.modelCollection:
            print(idx.getModelEntity().getModelInstanceName())


    def packModelForTrain(self, ready_pack_model_collection):
        packedModelTrainUnit = []
        for idx in ready_pack_model_collection:
            modelEntity = idx.getModelEntity()
            self.modelEntityCollection.append(modelEntity)
            modelLogit = modelEntity.build(self.features)
            self.logitCollection.append(modelLogit)
            modelCrossEntropy = modelEntity.cost(modelLogit, self.labels)
            self.crossEntropyCollection.append(modelCrossEntropy)
            with tf.name_scope('optimizer_packed'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    modelTrainStep = tf.train.AdamOptimizer(1e-4).minimize(modelCrossEntropy)
                    packedModelTrainUnit.append(modelTrainStep)

        return packedModelTrainUnit


    def singleModelForTrain(self, single_pack_model):
        singleModelTrainUnit = []
        modelEntity = single_pack_model.getModelEntity()
        self.modelEntityCollection.append(modelEntity)
        modelLogit = modelEntity.build(self.features)
        self.logitCollection.append(modelLogit)
        modelCrossEntropy = modelEntity.cost(modelLogit, self.labels)
        self.crossEntropyCollection.append(modelCrossEntropy)
        with tf.name_scope('optimizer_single'):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                modelTrainStep = tf.train.AdamOptimizer(1e-4).minimize(modelCrossEntropy)
                singleModelTrainUnit.append(modelTrainStep)

        return singleModelTrainUnit



                