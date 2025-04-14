import torch
import numpy as np
import math


class Loggers:
    def __init__(self, logger_path):
        self.logger_path = logger_path

    def __call__(self, input):
        input = str(input)
        with open(self.logger_path, 'a') as f:
            f.writelines(input + '\n')
        print(input)



def fake_label_record(fake_label, index, flag=True, updata=False, dataset='Regdb', trial=1):
    j = 0
    if dataset == 'RegDB':
        if flag:
            file = dataset + '/idx/train_visible_' + str(trial) + '_ul.txt'
        else:
            file = dataset + '/idx/train_thermal_' + str(trial) + '_ul.txt'
    elif dataset == 'SYSU-MM01':
        if flag:
            file = dataset + '/idx/train_visible_ul.txt'
        else:
            file = dataset + '/idx/train_thermal_ul.txt'
    f = open(file, "r", encoding='utf-8')
    lines = f.readlines()
    for i in index:
        local = lines[i].find('+')

        if local != -1:
            if not updata:
                continue
            else:

                lines[i] = lines[i][:local + 2] + str(fake_label[j].item()) + "\n"
        else:
            lines[i] = lines[i].replace("\n", ' ') + '+ ' + str(fake_label[j].item()) + "\n"
        j += 1
    f.close()

    with open(file, "w", encoding='utf-8') as ff:
        for i in range(len(lines)):
            ff.write(lines[i])
    ff.close()



def fake_label_read(index, flag=True):
    if flag:
        file = 'RegDB/train_visible_1_ul.txt'
    else:
        file = 'RegDB/train_thermal_1_ul.txt'
    f = open(file, "r", encoding='utf-8')
    lines = f.readlines()
    fake_label = []
    for i in index:
        local = lines[i].find('+')
        if local == -1:
            flabel = -1
            fake_label.append(flabel)
        else:
            end = lines[i].find('\n')
            flabel = int(lines[i][local + 2:end])
            fake_label.append(flabel)
    f.close()

    move_index = [i for i, x in enumerate(fake_label) if x == -1]

    if fake_label.count(-1) != 0:
        for i in range(fake_label.count(-1)):
            remove_index = fake_label.index(-1)

            del fake_label[remove_index]
    fake_label = torch.tensor(fake_label)
    return fake_label, index, move_index



def centre_record(feat, index, filename, dataset, flag=True, momentum=0.5):
    index = index.tolist()
    list1 = list(set(index))
    list2 = []
    zeros = torch.zeros(256)
    if dataset == 'RegDB':
        if flag:
            ave = torch.load('RegDB/visible_centre_' + str(filename) + '.pth')
        else:
            ave = torch.load('RegDB/thermal_centre_' + str(filename) + '.pth')
    elif dataset == 'SYSU-MM01':
        if flag:
            ave = torch.load('SYSU-MM01/visible_centre_' + str(filename) + '.pth')
        else:
            ave = torch.load('SYSU-MM01/thermal_centre_' + str(filename) + '.pth')
    for j in list1:
        idx = [i for i, x in enumerate(index) if x == j]
        list2.append(idx)


    times = len(list2)
    for i in range(times):
        for j in range(len(list2[i])):
            ave[list1[i]] = ave[list1[i]] * momentum + feat[list2[i][j]] * (1. - momentum)

    if flag:
        torch.save(ave, str(dataset) + '/visible_centre_' + str(filename) + '.pth')
    else:
        torch.save(ave, str(dataset) + '/thermal_centre_' + str(filename) + '.pth')


def centre_record_all(feat, index, dataset, momentum=0.5):
    index = index.tolist()
    list1 = list(set(index))
    list2 = []
    zeros = torch.zeros(256 * 6)

    if dataset == 'RegDB':
        ave = torch.load('RegDB/visible_centre_all.pth')
    elif dataset == 'SYSU-MM01':
        ave = torch.load('SYSU-MM01/visible_centre_all.pth')

    for j in list1:
        idx = [i for i, x in enumerate(index) if x == j]
        list2.append(idx)
    time_flag = torch.ones(206)
    for t in range(206):
        if ave[t].equal(zeros):
            time_flag[t] = 0

    times = len(list2)
    for i in range(times):
        for j in range(len(list2[i])):
            ave[list1[i]] = ave[list1[i]] * momentum + feat[list2[i][j]] * (1. - momentum)

    torch.save(ave, str(dataset) + '/visible_centre_all.pth')



def dis_calculation(feature, category_centre):
    n = feature.size(0)
    m = category_centre.size(0)
    Cen_dis = torch.zeros((n, m))
    for i in range(n):
        x = feature[i]
        for j in range(m):
            y = category_centre[j]
            d = torch.dist(x, y, p=2)
            Cen_dis[i, j] = d
    return Cen_dis



def dis_cos_cal(feature, category_centre):
    n = feature.size(0)
    m = category_centre.size(0)
    Cen_dis = torch.zeros((n, m))
    for i in range(n):
        x = feature[i]
        for j in range(m):
            y = category_centre[j]
            d = np.dot(x, y) / ((np.linalg.norm(x) * np.linalg.norm(y)) + 10e-6)
            Cen_dis[i, j] = d
    return Cen_dis



def fake_label_move(index, flag, dataset, trial):
    add_content = []
    add_content_u = []
    repeat_index = []
    add_content_back = []
    re_index = index

    if dataset == 'RegDB':
        if flag:
            file = dataset + '/idx/train_visible_' + str(trial) + '_ul.txt'
            wfile = dataset + '/idx/train_visible_' + str(trial) + '_l.txt'
            refer_file = dataset + '/train_visible_ul.txt'
        else:
            file = dataset + '/idx/train_thermal_' + str(trial) + '_ul.txt'
            wfile = dataset + '/idx/train_thermal_' + str(trial) + '_l.txt'
            refer_file = dataset + '/train_thermal_ul.txt'
    elif dataset == 'SYSU-MM01':
        if flag:
            file = dataset + '/idx/train_visible_ul.txt'
            wfile = dataset + '/idx/train_visible_l.txt'
            refer_file = dataset + '/train_visible_ul.txt'
        else:
            file = dataset + '/idx/train_thermal_ul.txt'
            wfile = dataset + '/idx/train_thermal_l.txt'
            refer_file = dataset + '/train_thermal_ul.txt'
    f = open(file, "r", encoding='utf-8')
    lines = f.readlines()
    lines_back = lines.copy()

    index = list(set(index))
    index = np.array(index)
    f.close()

    for i in index:
        add_content_u.append(lines[i])
    f = open(refer_file, "r", encoding='utf-8')
    lines_u = f.readlines()
    add_content_new = [i for i in add_content_u if i not in lines_u]
    repeat = [i for i in lines_u if i in add_content_u]
    for i in repeat:
        repeat_index.append(add_content_u.index(i))
    for i in sorted(repeat_index)[::-1]:
        del re_index[i]
    f.close()


    lines_u.extend(add_content_new)
    with open(refer_file, "w", encoding='utf-8') as ff:
        for i in range(len(lines_u)):
            ff.write(lines_u[i])


    for i in re_index:
        local_0 = lines[i].find('+')

        if local_0 != -1:
            local_1 = lines[i].find('.')
            local_2 = lines[i].find('\n')
            local_fake = lines[i][local_0 + 2:local_2]
            lines[i] = lines[i][:local_1 + 5] + local_fake + "\n"
        else:
            continue
        add_content.append(lines[i])


    f = open(wfile, "r", encoding='utf-8')
    lines = f.readlines()
    lines.extend(add_content)
    f.close()
    with open(wfile, "w", encoding='utf-8') as ff:
        for i in range(len(lines)):
            ff.write(lines[i])


    for i in index:
        local_0 = lines_back[i].find('+')

        if local_0 != -1:
            lines_back[i] = lines_back[i][:local_0 - 1] + "\n"
        else:
            continue
        add_content_back.append(lines_back[i])
    with open(file, "w", encoding='utf-8') as ff:
        for i in range(len(lines_back)):
            ff.write(lines_back[i])



def delet_process(index1, index2):
    print("Process delet function!")
    sort_index1 = list(set(index1))
    sort_index1.sort()
    sort_index2 = list(set(index2))
    sort_index2.sort()

    file = 'RegDB/idx/train_visible_1_ul.txt'
    f = open(file, "r", encoding='utf-8')
    lines = f.readlines()
    for i in reversed(sort_index1):
        del lines[i]
    f.close()
    with open(file, "w", encoding='utf-8') as ff:
        for i in range(len(lines)):
            ff.write(lines[i])
    ff.close()

    file = 'RegDB/idx/train_thermal_1_ul.txt'
    f = open(file, "r", encoding='utf-8')
    lines = f.readlines()
    for i in reversed(sort_index2):
        del lines[i]
    f.close()
    with open(file, "w", encoding='utf-8') as ff:
        for i in range(len(lines)):
            ff.write(lines[i])
    ff.close()



def del_process(location, dataset, c_t, trial):
    if c_t == 1:
        c_t = 'visible'
    else:
        c_t = 'thermal'
    if dataset == 'sysu':
        dataset = 'SYSU-MM01'
        file = dataset + '/idx/train_' + c_t + '_l.txt'
    else:
        dataset = 'RegDB'
        file = dataset + '/idx/train_' + c_t + '_' + str(trial) + '_l.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
        lens = (int(len(lines)) - location) * 0.1
        for i in range(math.floor(int(lens))):
            del lines[location]
    f.close()
    with open(file, 'w') as f:
        for line in lines:
            f.write(str(line))
    f.close()


def del_process2(dataset, c_t, trial):
    if c_t == 1:
        c_t = 'visible'
    else:
        c_t = 'thermal'
    if dataset == 'sysu':
        dataset = 'SYSU-MM01'
        file = dataset + '/train_' + c_t + '_ul.txt'
    else:
        dataset = 'RegDB'
        file = dataset + '/train_' + c_t + '_ul.txt'
    with open(file, 'r') as f:
        lines = f.readlines()
        lens = (int(len(lines))) * 0.1
        for i in range(math.floor(int(lens))):
            del lines[0]
    f.close()
    with open(file, 'w') as f:
        for line in lines:
            f.write(str(line))
    f.close()


def save_memory(feature, index, memory, flag, momentum, move):

    if flag:
        for i in range(feature.shape[0]):
            memory[index[i]] = memory[index[i]] * momentum + feature[i] * (1 - momentum)

    else:
        for i in range(feature.shape[0]):

            memory[index[i]] = memory[index[i]] * momentum + feature[i] * (1 - momentum)
    return memory
