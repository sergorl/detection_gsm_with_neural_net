from utility import util_fun as f
from utility import neuron as n


# ---------------------------------- Открытие размеченной выборки и графический вывод  ------------------------------- #
data = f.open_pickle("./train_sets/data_label_900MHz.pickle")
f.graph_out(data["data"], data["label"], step=1, thresh=0.6)


# ---------------------------------------------- Тренировка сети ----------------------------------------------------- #
dict_net = n.train_net(data["data"], data["label"], size_train=200, num_steps=1e3, rate_learn=0.5, init_rate=0.1,
                                                    file_for_save="./nets/net_1")



