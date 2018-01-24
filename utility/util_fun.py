import numpy as np
from numpy.matlib import repmat
import pickle
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.widgets import SpanSelector
import matplotlib.transforms as mtransforms


#-------------------------------------------- Создание меток для сигнала ----------------------------------------------#
def make_label_for_data(data, step, file_name):

    num_batch, size_batch = data.shape

    # Матрица меток:
    label = np.zeros_like(data)

    fig = plt.figure(figsize=(26, 8))

    line_data,  = plt.plot(data[0, :], ls='-', c='b', lw=1, label="spectrum")
    line_label, = plt.plot(data[0, :] * label[0, :], ls='--', c='g', lw=2, label="detected")

    ax = fig.get_axes()[0]

    plt.xlabel("samples")
    plt.ylabel("normalized spectrum, [0...1]")
    plt.grid(True)
    plt.legend()
    plt.xlim([0, size_batch])
    plt.tight_layout()

    class Index(object):
        ind = 0

        def draw_line(self):
            line_data.set_ydata(data[self.ind])
            line_label.set_ydata(data[self.ind]*label[self.ind, :])
            plt.draw()

        def next(self, event):
            self.ind = (self.ind + step) % num_batch
            ax.collections.clear()
            del ax.lines[2:]
            self.draw_line()

        def prev(self, event):
            self.ind = (self.ind - step) % num_batch
            ax.collections.clear()
            del ax.lines[2:]
            self.draw_line()

        def del_fill(self, event):
            if ax.collections:
                ax.collections.pop()
                l = ax.lines
                if len(l) > 2:
                    y_data = l[-1].get_ydata()
                    label[self.ind, y_data > 0] = 0
                    del l[-1]
                plt.draw()



    callback = Index()

    axfill = plt.axes([0.745, 0.93, 0.05, 0.055])
    axprev = plt.axes([0.8, 0.93, 0.05, 0.055])
    axnext = plt.axes([0.855, 0.93, 0.05, 0.055])

    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)

    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    bfill = Button(axfill, 'del fill')
    bfill.on_clicked(callback.del_fill)

    # Для SpanSelector'а
    trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

    # Функция заполнения цветом выделенной области и получение индексов заполнения:
    def onselect(xmin, xmax):
        ind_1                  = callback.ind
        ind_2                  = range(int(xmin), int(xmax)+1)
        label_select           = np.zeros((1, data.shape[1]))
        label_select[0, ind_2] = 1
        label[ind_1, ind_2]    = 1

        ax.plot(data[ind_1, :]*label_select[0, :], ls='--', c='g', lw=2, label="detected")

        ax.fill_betweenx(np.asarray([0, 1]), xmin, xmax, facecolor='green', alpha=0.3, transform=trans)
        fig.canvas.draw()


    # Выделение прямоугольной области курсором
    span = SpanSelector(ax, onselect, 'horizontal', useblit=True, rectprops=dict(alpha=0.5, facecolor='red'))

    # Показать графическое окно
    plt.show()

    # Выборочные данные:
    data_set  = np.zeros((1, data.shape[1]))
    label_set = np.zeros((1, data.shape[1]))

    for i in range(data.shape[0]):
        if label[i, :].sum() != 0:
            data_set  = np.vstack( (data_set,   data[i, :]) )
            label_set = np.vstack( (label_set, label[i, :]) )

    data_set  = np.delete(data_set, (0), axis=0)
    label_set = np.delete(label_set, (0), axis=0)

    # Сохранение
    if file_name != "":
        save_variable({"data": data_set, "label": label_set}, file_name)
        print("Данные сохранены в %s" % file_name)

    return data_set, label_set
#----------------------------------------------------------------------------------------------------------------------#


#------------------------------------- Графический вывод сигнала и меток ----------------------------------------------#
def graph_out(data, label, step, thresh):

    label[label > thresh] = 1
    label[label <= thresh] = 0

    num_batch, size_batch = data.shape

    plt.figure(figsize=(26, 8))
    line_data,  = plt.plot(data[0, :],              ls='-', c='b', lw=1, label="spectrum")
    line_label, = plt.plot(data[0, :]*label[0, :], ls='--', c='g', lw=2, label="detected")

    plt.xlabel("samples")
    plt.ylabel("normalized spectrum, [0...1]")
    plt.grid(True)
    plt.legend()
    plt.xlim([0, size_batch])
    plt.tight_layout()

    class Index(object):
        ind = 0

        def draw_line(self):
            line_data.set_ydata(data[self.ind])
            line_label.set_ydata(data[self.ind] * label[self.ind, :])
            plt.draw()

        def next(self, event):
            self.ind = (self.ind + step) % num_batch
            self.draw_line()

        def prev(self, event):
            self.ind = (self.ind - step) % num_batch
            self.draw_line()

    callback = Index()
    axprev = plt.axes([0.8, 0.93, 0.05, 0.055])
    axnext = plt.axes([0.855, 0.93, 0.05, 0.055])
    bnext = Button(axnext, 'Next')
    bnext.on_clicked(callback.next)
    bprev = Button(axprev, 'Previous')
    bprev.on_clicked(callback.prev)

    plt.show()
# ----------------------------------------------------------------------------------------------------------------------#



# --------------------------- Открытие нескольких файлов и объединение данных в одну матрицу---------------------------#
def open_file_list(file_names, size, type, opt):
    data = np.zeros((1, size))
    for name in file_names:
        data = np.vstack( (data, feature_normalization( open_binfile(name, size, type, opt)) ))
    return np.delete(data, (0), axis=0)
#----------------------------------------------------------------------------------------------------------------------#



#---------------------------------------------- Открытие бинарного файла ----------------------------------------------#
def open_binfile(name, size, type, opt):
    '''
    Возвращает вектор значений сигнала
    '''

    if type == 'signal':
        file = open(name, 'rb')
        signal = np.fromfile(file, dtype='float32')
    elif type == 'label':
        file = open(name, 'rb')
        signal = np.fromfile(file, dtype='uint32').astype(float)
    file.close()

    nums = int(len(signal) / size)
    # print("Число реализаций: " + str(nums))

    if opt == 'same':
        signal = signal[0:size]
    elif opt == 'average':
        buf          = signal.reshape((nums, size)).sum(axis = 0)/nums
        signal       = np.zeros((1, size))
        signal[0, :] = buf
    elif opt == 'full':
        signal = signal.reshape((nums, size))

    return signal
#----------------------------------------------------------------------------------------------------------------------#



#--------------------------------------- Сохранение параметров сети в формате .pickle ---------------------------------#
def save_variable(dict_of_variable, file_name):
    '''
    Сохраняет словарь с параметрами сети в файл с расширением .pickle
    '''
    with open(file_name, "wb") as f:
        pickle.dump(dict_of_variable, f)
# ---------------------------------------------------------------------------------------------------------------------#



#---------------------------------------------- Открытие файла .pickle ------------------------------------------------#
def open_pickle(file_name):
    '''
    Читает из файла
    :param file_name:
    :return:
    '''
    with open(file_name, 'rb') as f:
        out = pickle.load(f)
    return out
#----------------------------------------------------------------------------------------------------------------------#



#----------------------------------------------------------------------------------------------------------------------#
def open_pickle_list(pickle_list, size):
    data = np.zeros((1, size))
    for name in pickle_list:
        data = np.vstack( (data, open_pickle(name)) )
    return np.delete(data, (0), axis=0)
#----------------------------------------------------------------------------------------------------------------------#



#----------------------------------- Выравнивание входных данных до требуемой размерности -----------------------------#
def reshapeof(x, remain, dimensions):
    '''
    Преобразование входного масива x в матрицу размерности dimensions
    '''
    num_block, len_block = dimensions
    if remain != 0:
        if len(x.shape) == 2:
            return np.hstack( (x, x[:, -(len_block-remain):]) ).reshape((num_block, len_block))
        else:
            return np.hstack((x, x[-(len_block-remain):])).reshape((num_block, len_block))
    else:
        return x.reshape((num_block, len_block))
# ---------------------------------------------------------------------------------------------------------------------#



# ------------------------------------------- Нормировка выборки ------------------------------------------------------#
def feature_normalization(x):

    mins = x.min(axis=1)
    down = mins.reshape((mins.size, 1))
    down = repmat(down, 1, x.shape[1])
    out  = x + down.__abs__()

    maxs = out.max(axis=1)
    up   = maxs.reshape((maxs.size, 1))
    up   = repmat(up, 1, x.shape[1])
    out  = out / up

    return out
# ---------------------------------------------------------------------------------------------------------------------#



# --------------------------------------- Центрирование и нормировка --------------------------------------------------#
def centr_normalization(data, dimensions):

    # Центрирование:
    num_block, len_block = dimensions
    data -= repmat(data.mean(axis=1).reshape((num_block, 1)), 1, len_block)

    # Нормировка:
    maxs  = data.std(axis=1)
    up    = maxs.reshape((maxs.size, 1))
    up    = repmat(up, 1, len_block)
    data /= up

    return data
# ---------------------------------------------------------------------------------------------------------------------#