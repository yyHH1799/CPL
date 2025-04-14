
f = open('train_visible_1.txt', "r", encoding='utf-8')
lines = f.readlines()
label_index = []
unlabel_index = []
j = 0
for i in range(len(lines)):
    if j < 2:
        label_index.append(lines[i])
        j += 1
    elif 1 < j < 10:
        unlabel_index.append(lines[i])
        j += 1
        if j == 10:
            j = 0
f.close()

with open('train_visible_1_l.txt', "w", encoding='utf-8') as ff:
    for i in range(len(label_index)):
        ff.write(label_index[i])
ff.close()
with open('train_visible_1_ul.txt', "w", encoding='utf-8') as ff:
    for i in range(len(unlabel_index)):
        ff.write(unlabel_index[i])
ff.close()


f = open('train_thermal_1.txt', "r", encoding='utf-8')
lines = f.readlines()
label_index = []
unlabel_index = []
j = 0
for i in range(len(lines)):
    if j < 2:
        label_index.append(lines[i])
        j += 1
    elif 1 < j < 10:
        unlabel_index.append(lines[i])
        j += 1
        if j == 10:
            j = 0
f.close()

with open('train_thermal_1_l.txt', "w", encoding='utf-8') as ff:
    for i in range(len(label_index)):
        ff.write(label_index[i])
ff.close()
with open('train_thermal_1_ul.txt', "w", encoding='utf-8') as ff:
    for i in range(len(unlabel_index)):
        ff.write(unlabel_index[i])
ff.close()