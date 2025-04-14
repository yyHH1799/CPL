import os
import random



def getDirList(p):
    p = str(p)
    if p == "":
        return []
    p = p.replace("/", "\\")
    if p[-1] != "\\":
        p = p + "\\"
    a = os.listdir(p)
    b = [x for x in a if os.path.isdir(p + x)]
    return b



def getFileList(p):
    p = str(p)
    if p == "":
        return []
    p = p.replace("/", "\\")
    if p[-1] != "\\":
        p = p + "\\"
    a = os.listdir(p)
    b = [x for x in a if os.path.isfile(p + x)]
    return b



def Random_Choice_File(fileDir, rate):
    pathDir = os.listdir(fileDir)
    filenumber = len(pathDir)
    rate = rate
    picknumber = int(filenumber * rate)
    sample = random.sample(pathDir, picknumber)
    unleabeled = []
    for file in pathDir:
        if file not in sample:
            unleabeled.append(file)
    return sample, unleabeled


if __name__ == '__main__':
    dir = 'SYSU-MM01/'
    labeled_rate = 0.2
    with open(dir + 'exp/train_id.txt', 'r') as f:
        trainlist = f.read().split(',')
    cam = getDirList(dir)
    leabeledlist = []
    unleabeledlist = []
    for i in range(1, 7):
        for realidx, trainidx in enumerate(trainlist):
            if int(trainidx) < 10:
                pnum = '000' + str(trainidx)
            elif 9 < int(trainidx) < 100:
                pnum = '00' + str(trainidx)
            else:
                pnum = '0' + str(trainidx)
            try:
                randomfile, unleabeled = Random_Choice_File(dir + 'cam' + str(i) + '/' + str(pnum), rate=labeled_rate)
                randomfile.sort()
                unleabeled.sort()
                for j in range(len(randomfile)):
                    leabeledlist.append('cam' + str(i) + '/' + str(pnum) + '/' + randomfile[j] + ' ' + str(realidx))
                for j in range(len(unleabeled)):
                    unleabeledlist.append('cam' + str(i) + '/' + str(pnum) + '/' + unleabeled[j] + ' ' + str(realidx))
            except:
                pass

    with open('train_visible_l.txt', 'w') as f:
        for i in range(len(leabeledlist)):
            if int(leabeledlist[i].split('/')[0].split('m')[1]) in (1, 2, 4, 5):
                f.write(leabeledlist[i] + '\n')

    with open('train_thermal_l.txt', 'w') as f:
        for i in range(len(leabeledlist)):
            if int(leabeledlist[i].split('/')[0].split('m')[1]) in (3, 6):
                f.write(leabeledlist[i] + '\n')

    with open('train_visible_ul.txt', 'w') as f:
        for i in range(len(unleabeledlist)):
            if int(unleabeledlist[i].split('/')[0].split('m')[1]) in (1, 2, 4, 5):
                f.write(unleabeledlist[i] + '\n')

    with open('train_thermal_ul.txt', 'w') as f:
        for i in range(len(unleabeledlist)):
            if int(unleabeledlist[i].split('/')[0].split('m')[1]) in (3, 6):
                f.write(unleabeledlist[i] + '\n')
