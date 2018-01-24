from utility import util_fun as f


# -------------------------------------- Формирование размеченной выборки  ------------------------------------------- #
data                = f.open_file_list(["./train_sets/set_900MHz.bin"], size=6150, type="signal", opt="full")

data_set, label_set = f.make_label_for_data(data, step=30, file_name="./train_sets/data_label_900MHz.pickle")





