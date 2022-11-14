import tflearn
from data_process import data_process


# # change philosopher name here
# philosopher_name = ""
# [words, labels, training, output] = data_process(
#     "intent", "pikl", philosopher_name)


def model_train(training, output, philosopher_name):

    net = tflearn.input_data(shape=[None, len(training[0])])
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(net, 8)
    net = tflearn.fully_connected(
        net, len(output[0]), activation="softmax")
    net = tflearn.regression(net)

    model = tflearn.DNN(net)
    model.fit(training, output, n_epoch=1000,
              batch_size=8, show_metric=True)

    return model
