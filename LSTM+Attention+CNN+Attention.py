# mnist attention
from __future__ import print_function
import numpy as np
np.random.seed(1337)
# import tensorflow as tf
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.layers.convolutional import Convolution1D, MaxPooling1D
from sklearn import metrics
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
import numpy as np
np.random.seed(1337)  # for reproducibility
tf.disable_v2_behavior()

import pickle
def trans(seq):
    a = []
    dic = {'A':1,'B':21,'U':25,'J':24,'Z':23,'O':26,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':9,'L':10,'M':11,'N':12,'P':13,'Q':14,'R':15,'S':16,'T':17,'V':18,'W':19,'Y':20,'X':22}
    for i in range(len(seq)):
        a.append(dic.get(seq[i]))
    return a

def createTrainData(fp):
    id_num = []
    sequence_num = []
    label_num = []
    for line in open(fp):
        proteinId, sequence, label = line.split(",")
        proteinId = proteinId.strip(' \t\r\n')
        id_num.append(proteinId)
        sequence = sequence.strip(' \t\r\n')
        sequence_num.append(trans(sequence))
        label = label.strip(' \t\r\n')
        label_num.append(int(label))
    return id_num,sequence_num, label_num


def createTrainTestData(str_path, test_split=0.2, seed=113):
    ids,X,labels = pickle.load(open(str_path, "rb"))
    np.random.seed(seed)
    np.random.shuffle(ids)
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(labels)

    id_train = np.array(ids[:int(len(X) * (1 - test_split))])
    X_train = np.array(X[:int(len(X) * (1 - test_split))])
    y_train = np.array(labels[:int(len(X) * (1 - test_split))])

    id_test = np.array(ids[int(len(X) * (1 - test_split)):])
    X_text = np.array(X[int(len(X) * (1 - test_split)):])
    y_test = np.array(labels[int(len(X) * (1 - test_split)):])

    return (id_train,X_train, y_train), (id_test,X_text, y_test)

def printTrainTestData(id_train,X_train, y_train,id_test, X_text, y_test):
    # print train set and test set
    dic = {1: 'A', 21: 'B', 25: 'U', 24: 'J', 23: 'Z', 26: 'O', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
           9: 'K', 10: 'L', 11: 'M', 12: 'N', 13: 'P', 14: 'Q', 15: 'R', 16: 'S', 17: 'T', 18: 'V', 19: 'W', 20: 'Y',
           22: 'X'}
    xtrain = []
    for i in range(len(X_train)):
        str1 = ''
        for j in range(maxlen):
            if (X_train[i][j] == 0):
                str1 += '0'
            else:
                str1 += dic.get(X_train[i][j])
        xtrain.append(str1)
    xtest = []
    for i in range(len(X_text)):
        str1 = ''
        for j in range(maxlen):
            if (X_text[i][j] == 0):
                str1 += '0'
                continue
            else:
                str1 += dic.get(X_text[i][j])
        xtest.append(str1)
    trainFile_output = open('trainfile.csv', 'w', encoding='gbk')
    testFile_output = open('testfile.csv', 'w', encoding='gbk')
    for i in range(len(xtrain)):
        trainFile_output.write(id_train[i] + ',' + xtrain[i] + ',' + str(y_train[i]) + '\n')
    for i in range(len(xtest)):
        testFile_output.write(id_test[i] + ',' + xtest[i] + ',' + str(y_test[i]) + '\n')

class Attention(Layer):
    def __init__(self, attention_size, **kwargs):
        self.attention_size = attention_size
        super(Attention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name="W_{:s}".format(self.name),
                                 shape=(input_shape[-1].value, self.attention_size),
                                 initializer="glorot_normal",
                                 trainable=True)
        self.b = self.add_weight(name="b_{:s}".format(self.name),
                                 shape=(input_shape[1].value, 1),
                                 initializer="zeros",
                                 trainable=True)
        self.u = self.add_weight(name="u_{:s}".format(self.name),
                                 shape=(self.attention_size, 1),
                                 initializer="glorot_normal",
                                 trainable=True)
        super(Attention, self).build(input_shape)
    def call(self, x, mask=None):
        et = K.tanh(K.dot(x, self.W) + self.b)
        at = K.softmax(K.squeeze(K.dot(et, self.u), axis=-1))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        atx = K.expand_dims(at, axis=-1)
        ot = atx * x
        output = K.sum(ot, axis=1)
        return output, at
    def compute_mask(self, input, input_mask=None):
        return None
    def compute_output_shape(self, input_shape):
         return (input_shape[0], input_shape[-1])
    def get_config(self):
        config = {
            'attention_size': self.attention_size,
        }
        base_config = super(Attention, self).get_config()
        return dict(list(config.items()))

def build_common_model(maxlen):
    # Embedding
    embedding_size = 256
    expend_dim = 24
    # LSTM
    units = 128
    # Attention
    attention_size = 2
    # Convolution
    filters = 128
    pool_size = 2
    input = Input(shape = (maxlen,), dtype='int32')
    x = Embedding(output_dim = embedding_size, input_dim = expend_dim , input_length = maxlen,
                  trainable = False, name = 'embedding')(input)
    x = Dropout(0.5)(x)
    lstm = LSTM(units = units, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(x)
    att1, att1_w = Attention(attention_size = attention_size, name = 'LSTM-Attention')(lstm)
    con1D = Convolution1D(filters=filters, kernel_size=10,padding='valid',activation='relu', strides=1)(x)
    po1 = MaxPooling1D(pool_size=pool_size)(con1D)
    con1D2 = Convolution1D(filters=filters,kernel_size=5,padding='valid',activation='relu',strides=1)(po1)
    po2 = MaxPooling1D(pool_size=pool_size)(con1D2)
    att2, att2_w = Attention(attention_size=attention_size, name='CNN-Attention')(po2)
    x = Concatenate()([att1, att2])
    x = Dense(64, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation = 'sigmoid')(x)
    model = Model(inputs = input, outputs = out)
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
def attention_layer_output(model,X_train):
    attention_layer_output_model = Model(inputs=model.input,
                                         outputs=model.get_layer('LSTM-Attention').output)
    _, attention_layer_output = attention_layer_output_model.predict(X_train)
    np.savetxt('attentionScore.txt', attention_layer_output, fmt='%f')
    print('attention_layer_output.shape:',attention_layer_output.shape)
def print_evaluation_measures(y_pred,X_text,y_test):
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    print('test auc:', auc)

    pre = model.predict(X_text)
    pre[np.where(pre < 0.5)] = 0
    pre[np.where(pre >= 0.5)] = 1
    pre = np.squeeze(pre)
    y_test = np.squeeze(y_test)
    pre_temp_index = (pre == y_test)
    pre_temp_true = pre[pre_temp_index]
    index = np.where(pre_temp_true == 1)
    TP = np.sum(pre_temp_true[index])
    print('TP is ', TP)

    pre_temp_index = (pre == y_test)
    pre_temp_true = pre[pre_temp_index]
    index = np.where(pre_temp_true == 0)
    temp = pre_temp_true[index]
    TN = len(temp)
    print('TN is ', TN)

    pre_temp_index = (pre != y_test)
    pre_temp_fals = pre[pre_temp_index]
    index = np.where(pre_temp_fals == 0)
    temp = pre_temp_fals[index]
    FN = len(temp)
    print('FN is', FN)

    pre_temp_index = (pre != y_test)
    pre_temp_fals = pre[pre_temp_index]
    index = np.where(pre_temp_fals == 1)
    FP = np.sum(pre_temp_fals[index])
    print('FP is ', FP)

    sensitivity = TP / np.maximum((TP + FN), 1)
    specificity = TN / np.maximum((TN + FP), 1)
    MCC = (TP * TN - FP * FN) / (((TP + FN) * (TN + FP) * (TP + FP) * (TN + FN)) ** 0.5)
    print('sensitivity is ', sensitivity)
    print('specificity is ', specificity)
    print('MCC',MCC)

if __name__ == "__main__":
    csvFile = 'smallBalanceData.csv'
    pklFile = 'smallBalanceData.pkl'
    a , b, c = createTrainData(csvFile)
    t = (a, b, c)
    pickle.dump(t, open(pklFile, "wb"))
    maxlen = 1280
    batch_size = 128
    nb_epoch = 1
    (id_train, X_train, y_train), (id_test, X_text, y_test) = createTrainTestData(pklFile)
    X_train = tf.keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
    X_text = tf.keras.preprocessing.sequence.pad_sequences(X_text, maxlen=maxlen)
    print('X_train shape:', X_train.shape)
    print('X_text shape:', X_text.shape)
    printTrainTestData(id_train,X_train, y_train, id_test,X_text, y_test)
    model = build_common_model(maxlen)
    print('summary()')
    print(model.summary())
    print('training ... ')

    model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch)
    model.save('model.h5')
    attention_layer_output(model, X_train)
    print('test ... ')
    loss, accuracy = model.evaluate(X_text, y_test)
    print('test loss:', loss)
    print('test accuracy:', accuracy)
    y_pred = model.predict(X_text, batch_size=batch_size)
    print_evaluation_measures(y_pred, X_text, y_test)
