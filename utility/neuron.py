import numpy as np
import time
import tensorflow as tf
from utility import util_fun as f

# --------------------------------------------------- Net-model -------------------------------------------------------#
def net_model(size_train):

    input = tf.placeholder(tf.float64, (None, size_train), name="input")
    right = tf.placeholder(tf.float64, (None, size_train))

    sigma, bias, k = 0.1, 0.5, 3

    W1 = tf.Variable(tf.truncated_normal((size_train,     k * size_train), stddev=sigma, dtype=tf.float64))
    W2 = tf.Variable(tf.truncated_normal((k * size_train, k * size_train), stddev=sigma, dtype=tf.float64))
    W3 = tf.Variable(tf.truncated_normal((k * size_train,     size_train), stddev=sigma, dtype=tf.float64))

    b1 = tf.Variable(bias * tf.ones((1, k * size_train), dtype=tf.float64))
    b2 = tf.Variable(bias * tf.ones((1, k * size_train), dtype=tf.float64))
    b3 = tf.Variable(bias * tf.ones((1,     size_train), dtype=tf.float64))

    y1      = tf.sin(tf.matmul(input, W1) + b1)
    y2      = tf.sin(tf.matmul(y1, W2) + b2)
    predict = tf.sigmoid(tf.matmul(y2, W3) + b3, name="predict")

    return input, right, predict
# ---------------------------------------------------------------------------------------------------------------------#


#----------------------------------------------- Обработка данных сетью  ----------------------------------------------#
def run_predict(data, file_name):
    '''
    Запуск сети для обработки данных по всем временным срезам
    '''
    out_predict = np.zeros_like(data) # выходы сети (результат работы модели)
    num_batch, size_batch = data.shape

    # Восстановление графа и модели
    graph = tf.Graph()

    with graph.as_default():

        saver   = tf.train.import_meta_graph(file_name + ".ckpt.meta")

        input   = graph.get_tensor_by_name("input:0")
        predict = graph.get_tensor_by_name("predict:0")

        size_win = input.get_shape()[1].__int__()  # размерность окна равна числу нейронов первого слоя

        # Формирование batch'ей:
        remain     = size_batch % size_win
        num_step   = size_batch // size_win
        dimensions = (num_step, size_win) if remain == 0 else (num_step + 1, size_win)

        with tf.Session() as sess:
            saver.restore(sess, file_name + ".ckpt")

            for i in range(num_batch):

                batch = f.centr_normalization(f.reshapeof(data[i, :], remain, dimensions), dimensions)

                out               = sess.run(predict, feed_dict={input: batch})
                out               = out.reshape((1, dimensions[0] * dimensions[1]))
                out_predict[i, :] = out[0,0:size_batch]

                print("Обработка ... %.2f %%" % (100*(i+1)/num_batch))

    return out_predict
#----------------------------------------------------------------------------------------------------------------------#



#--------------------------------------- Формирование выборки требуемой размерности -----------------------------------#
def make_data(data, labels, size_set):

    dims = data.shape
    if len(dims) == 2:
        nums, length = dims
    else:
        nums, length = 1, data.size

    remain = length % size_set
    num_block_in_batch = length // size_set

    dimensions = (nums * num_block_in_batch, size_set) if remain == 0 else (nums * (num_block_in_batch + 1), size_set)

    batch_train  = f.reshapeof(data, remain, dimensions)
    batch_labels = f.reshapeof(labels, remain, dimensions)

    shift        = np.random.randint(length)

    batch_train  = np.roll(batch_train, shift, axis=1)
    batch_train += 0.01*np.random.randn(batch_train.shape[0], batch_train.shape[1])

    # Центрирование и нормировка
    batch_train = f.centr_normalization(batch_train, dimensions)

    # Циклический сдвиг
    batch_labels = np.roll(batch_labels, shift, axis=1)

    return batch_train, batch_labels
#----------------------------------------------------------------------------------------------------------------------#



#------------------------------------------------- Тренировка сети ----------------------------------------------------#
def train_net(data, labels, size_train, num_steps, rate_learn, init_rate, file_for_save):
    '''
    Тренировка сети по обучающей выборке
    size_train - размер одной выборки (размер одной реализациия спектра)
    '''

    losses  = []  # список значений функции потерь

    input, right, predict = net_model(size_train) # модель
    loss                  = tf.reduce_mean((predict - right) ** 2) # функция потерь

    # Градиентные методы оптимизации:
    train_step = tf.train.AdagradOptimizer(rate_learn, initial_accumulator_value=init_rate).minimize(loss)
    # train_step = tf.train.GradientDescentOptimizer(rate_learn).minimize(loss)
    # train_step = tf.train.AdadeltaOptimizer(rate_learn, rho=0.9).minimize(loss)

    # Для сохранения модели:
    saver = tf.train.Saver()

    start  = time.time()

    # Тренировка
    with tf.Session() as sess:

        sess.run(tf.initialize_all_variables())

        for i in range(int(num_steps)):
            batch_train, batch_labels = make_data(data, labels, size_train)

            los, _  = sess.run([loss,train_step], feed_dict={input: batch_train, right: batch_labels})
            losses += [los]

            print("Потери: %.6f ... прогресс: %.2f %%" % (los, 100*(i + 1)/num_steps))

        end = time.time()

        # Сохранение параметров сети:
        save_path = saver.save(sess, file_for_save + ".ckpt", meta_graph_suffix='meta', write_meta_graph=True)
        print("Model saved in file: %s" % save_path)

    print("Тренировка пройдена за %.2f c" % (end - start))

    # Тест по тренировочным данным
    predict = run_predict(data, file_for_save)
    f.graph_out(data, predict, step=1, thresh=0.5)

    return losses
#----------------------------------------------------------------------------------------------------------------------#
