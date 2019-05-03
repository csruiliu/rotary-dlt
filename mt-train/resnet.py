import tensorflow as tf

class ResNet(object):
    def __init__(self):
        pass

    def conv_layer(self):
        pass

    def build(self,input):
        print("building resnet...")
        x = tf.pad(input, tf.constant([[0, 0], [3, 3, ], [3, 3], [0, 0]]), "CONSTANT")
        assert(x.shape == (x.shape[0],70,70,3))
        print("build resnet successufully")

    def train(self):
        features = tf.placeholder(tf.float32, [None, 64, 64, 3])
        self.build(features)
        print("training...")

def main(_):

    resnet = ResNet()
    resnet.train()

if __name__ == '__main__':
    tf.app.run(main=main)
