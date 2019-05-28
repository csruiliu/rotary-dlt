import tensorflow as tf
from timeit import default_timer as timer


img_w = 224
img_h = 224

class Schedule(object):
    def __init__(self, model_collection):
        self.modelCollection = model_collection
        self.modelEntityCollection = []
        self.logitCollection = []
        self.crossEntropyCollection = []
        self.trainStepColllection = []
        self.scheduleCollection = []

        self.features = tf.placeholder(tf.float32, [None, img_h, img_w, 3])
        self.labels = tf.placeholder(tf.int64, [None, 1000])

    def showAllModels(self):
        for idx in self.modelCollection:
            print(idx.getModelName())

    def pack(self, ready_pack_model_collection):
        #packedModelCollection = []
        for idx in ready_pack_model_collection:
            modelEntity = idx.getModelEntity()
            self.modelEntityCollection.append(idx.getModelEntity())
            modelLogit = modelEntity.build(self.features)
            self.logitCollection.append(modelLogit)
            modelCrossEntropy = modelEntity.cost(modelLogit, self.labels)
            self.crossEntropyCollection.append(modelCrossEntropy)
            with tf.name_scope('optimizer'):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    modelTrainStep = tf.train.AdamOptimizer(1e-4).minimize(modelCrossEntropy)
                    self.trainStepCollection.append(modelTrainStep)

        #return packedModelCollection

    def schedule(self):
        scheduleUnit1 = []
        scheduleUnit.append()
        self.scheduleCollection.append(scheduleUnit)


    def executeSch(self, X_train, Y_train):
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
                for trIdex in self.scheduleCollection:


                    sess.run(self.trainStepColllection, feed_dict={self.features: X_mini_batch_feed, self.labels: Y_mini_batch_feed})
                end_time = timer()
                total_time += end_time - start_time
            print("training time for 1 epoch:", total_time)