import numpy as np
from PIL import Image
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):
        data_dir = data_dir

        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')


        self.train_color_image = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):
        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class SYSUData_u(data.Dataset):
    def __init__(self, data_dir, transform=None, rIndex1=None, rIndex2=None):
        data_dir = data_dir

        train_color_image_u = np.load(data_dir + 'train_rgb_resized_img_unlabeled.npy')
        train_thermal_image_u = np.load(data_dir + 'train_ir_resized_img_unlabeled.npy')
        self.train_color_image_u = train_color_image_u
        self.train_thermal_image_u = train_thermal_image_u


        self.transform = transform

        self.rIndex1 = rIndex1
        self.rIndex2 = rIndex2

    def __getitem__(self, index):
        img3 = self.train_color_image_u[self.rIndex1[index]]
        img4 = self.train_thermal_image_u[self.rIndex2[index]]

        img3 = self.transform(img3)
        img4 = self.transform(img4)

        return img3, img4

    def __len__(self):
        return len(self.train_color_image_u)


class sysuData(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex=None, thermalIndex=None):

        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_l.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_l.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)


        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class sysuData_u(data.Dataset):
    def __init__(self, data_dir, transform=None, colorIndex_u=None, thermalIndex_u=None):

        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_ul.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_ul.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)

        self.train_color_image_u = train_color_image
        self.train_color_label_u = train_color_label

        self.train_thermal_image_u = train_thermal_image
        self.train_thermal_label_u = train_thermal_label

        self.transform = transform
        self.cIndex_u = colorIndex_u
        self.tIndex_u = thermalIndex_u

    def __getitem__(self, index):
        img3, target3 = self.train_color_image_u[self.cIndex_u[index]], self.train_color_label_u[self.cIndex_u[index]]
        img4, target4 = self.train_thermal_image_u[self.tIndex_u[index]], self.train_thermal_label_u[self.tIndex_u[index]]

        img3 = self.transform(img3)
        img4 = self.transform(img4)

        return img3, img4, target3, target4

    def __len__(self):
        return len(self.train_color_label_u)


class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex=None, thermalIndex=None, colorIndex_u=None,
                 thermalIndex_u=None):

        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '_l.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '_l.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)


        self.train_color_image = train_color_image
        self.train_color_label = train_color_label

        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label

        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1, target1 = self.train_color_image[self.cIndex[index]], self.train_color_label[self.cIndex[index]]
        img2, target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]

        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)


class RegDBData_u(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex_u=None, thermalIndex_u=None):

        data_dir = data_dir
        train_color_list = data_dir + 'idx/train_visible_{}'.format(trial) + '_ul.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial) + '_ul.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)

        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir + color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image)

        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir + thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)


        self.train_color_image_u = train_color_image
        self.train_color_label_u = train_color_label

        self.train_thermal_image_u = train_thermal_image
        self.train_thermal_label_u = train_thermal_label

        self.transform = transform
        self.cIndex_u = colorIndex_u
        self.tIndex_u = thermalIndex_u

    def __getitem__(self, index):
        img3, target3 = self.train_color_image_u[self.cIndex_u[index]], self.train_color_label_u[self.cIndex_u[index]]
        img4, target4 = self.train_thermal_image_u[self.tIndex_u[index]], self.train_thermal_label_u[self.tIndex_u[index]]

        img3 = self.transform(img3)
        img4 = self.transform(img4)

        return img3, img4, target3, target4

    def __len__(self):
        return len(self.train_color_label_u)




class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


class TestDataOld(data.Dataset):
    def __init__(self, data_dir, test_img_file, test_label, transform=None, img_size=(144, 288)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(data_dir + test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1, target1 = self.test_image[index], self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)


def load_data(input_data_path):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()

        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]

    return file_image, file_label
