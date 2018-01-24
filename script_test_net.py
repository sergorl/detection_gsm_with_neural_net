from utility import util_fun as f
from utility import neuron as n


#--------------------------------------------------- Данные -----------------------------------------------------------#
data_test = f.open_file_list(["./train_sets/set_900MHz.bin"], size=6150, type="signal", opt="full")
data_test = f.feature_normalization(data_test)


#------------------------------------------------ Тестирование --------------------------------------------------------#
predict = n.run_predict(data_test, "./nets/net_1")


#---------------------------------------------- Вывод результатов -----------------------------------------------------#
f.graph_out(data_test, predict, step=10, thresh=0.5)



