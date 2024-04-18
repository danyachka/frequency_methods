import pandas as pd
import numpy as np
import os
from scipy import signal
import matplotlib.pyplot as plt


def readCsv(file_suf, dateFormat):
    path = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "\\data\\lab4\\"
    file_path = path + f'sber_{file_suf}.csv'
    print(file_path)
    data = pd.read_csv(file_path, delimiter=";")

    data = data.drop("<TICKER>", axis=1)
    data = data.drop("<PER>", axis=1)

    data["<DATE>"] = data["<DATE>"].astype(str)
    data["<TIME>"] = data["<TIME>"].astype(str)

    data["<DATE>"] = data["<DATE>"].str.cat(data["<TIME>"], sep=" ")
    print(data["<DATE>"])
    data["<DATE>"] = pd.to_datetime(data["<DATE>"], format=dateFormat)

    data.set_index('<DATE>', inplace=True)

    return data


def apply_filter(data, time_constant):
    b = np.poly1d([1])
    a = np.poly1d([1 + time_constant, -time_constant])

    # filter_ = signal.TransferFunction([1], [time_constant, 1])
    # _, y1, _ = signal.lsim(filter_, U=data['<CLOSE>'], T=data.index.astype("int64"))
    # return filter_

    zi = signal.lfilter_zi(b, a)
    filtered, _ = signal.lfilter(b, a, data, zi=zi * data[0])
    # filtered, _ = signal.lfilter(b, a, filtered, zi=zi*filtered[0])

    # filtered = signal.filtfilt(b, a, data)
    return filtered


def main():
    data_4y = readCsv("4y", "%y%m%d %H")
    data_4m = readCsv("4m", "%y%m%d %H%M%S")

    time_constants = [1 * 14, 7 * 14, 30, 90, 365]
    dataList = {time_constants[0]: data_4m,
                time_constants[1]: data_4m,
                time_constants[2]: data_4y,
                time_constants[3]: data_4y,
                time_constants[4]: data_4y}

    TTags = {time_constants[0]: "День",
             time_constants[1]: "Неделя",
             time_constants[2]: "Месяц",
             time_constants[3]: "3 месяца",
             time_constants[4]: "Год"}

    for idx, time_constant in enumerate(time_constants, 1):
        plt.figure(figsize=(12, 8))
        data = dataList[time_constant]
        tag = TTags[time_constant]
        filtered_data = apply_filter(data["<CLOSE>"], time_constant)

        # plt.subplot(len(time_constants), 1, idx)
        plt.plot(data.index, data['<CLOSE>'], label='Исходные данные', color='blue', alpha=0.7)

        plt.plot(data.index, filtered_data, label=f'Фильтр, T = {tag}', color='red')

        plt.title(f'Фильтрация с T = {tag}')
        plt.xlabel('Дата')
        plt.ylabel('Цена закрытия')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()
