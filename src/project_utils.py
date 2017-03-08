import numpy as np
from PIL import Image
import os
import random as rd
from scipy import ndimage
from shutil import copyfile
import shutil


def resize_image(scale):
    folder_path = '/home/gabi/Documents/datasets/caltech-faces/'
    num = 1
    if folder_path.endswith('/'):
        num = 2

    parts = folder_path.split('/')
    new_path = ''
    for i in range(0, len(parts) - num):
        new_path = os.path.join(new_path, parts[i])

    new_path = os.path.join('/', new_path, 'caltech-faces-small')

    if not os.path.exists(new_path):
        print('creating new folder')
        os.makedirs(new_path)

    images = os.listdir(folder_path)

    for index in range(0, len(images)):
        img_name = os.path.join(folder_path, images[index])
        img = Image.open(img_name)
        new_img_w, new_img_h = img.size
        new_img_w = new_img_w / 2
        new_img_h = new_img_h / 2
        new_img = img.resize((new_img_w, new_img_h))
        new_img_name = os.path.join(new_path, ('face_%d.jpg' %index))
        new_img.save(new_img_name, 'JPEG')

    # TODO: fix order of saved images, remove readme file


# generate noise to test if siamese_cnn pipeline is working
def load_data():
    print('loading data')
    train = np.array([])
    validate = np.array([])
    for num in range(0, pc.AMOUNT_DATA):
        print(num)
        imarray = np.random.rand(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS) * 255
        im_t_1 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        im_t_2 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        train = np.append(train, [im_t_1, im_t_2])
        imarray = np.random.rand(pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS) * 255
        im_v_1 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        im_v_2 = np.asarray(Image.fromarray(imarray.astype('uint8')).convert('RGB'))
        validate = np.append(validate, [im_v_1, im_v_2])

    train = train.reshape([pc.AMOUNT_DATA, pc.NUM_CLASSES, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
    validate = validate.reshape([pc.AMOUNT_DATA, pc.NUM_CLASSES, pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS])
    ans = [train, validate]
    return ans


# crop images in center
def crop_images(folder_path, width, height):
    num = 1
    if folder_path.endswith('/'):
        num = 2

    parts = folder_path.split('/')
    new_path = ''
    for i in range(0, len(parts)-num):
        new_path = os.path.join(new_path, parts[i])


    list_images = os.listdir(folder_path)
    name_folder = folder_path.split('/')[-num]
    new_folder_path = os.path.join(new_path, 'real_cropped_images_' + str(name_folder))
    new_folder_path = '/' + new_folder_path
    if not os.path.exists(new_folder_path):
        print('asdf')
        os.makedirs(new_folder_path)

    for image_path in list_images:
        img = Image.open(os.path.join(folder_path, image_path))
        img_width, img_height = img.size


        center_x = img_width / 2
        center_y = img_height / 2
        start_x = center_x - width / 2
        start_y = center_y - height / 2
        img2 = img.crop((start_x, start_y, start_x + width, start_y + height))
        img2.save(os.path.join(new_folder_path, image_path))

    pass


def load_INRIA_data(path):

    print('loading data')
    # load into lists
    train_data_names_pos_ = np.genfromtxt(os.path.join(path, 'train_data_names_pos.csv'), dtype=None).tolist()
    train_data_names_neg_ = np.genfromtxt(os.path.join(path, 'train_data_names_neg.csv'), dtype=None).tolist()
    train_data_labels_pos_ = np.genfromtxt(os.path.join(path, 'train_data_labels_pos.csv'), dtype=None).tolist()
    train_data_labels_neg_ = np.genfromtxt(os.path.join(path, 'train_data_labels_neg.csv'), dtype=None).tolist()
    validation_data_names_pos_ = np.genfromtxt(os.path.join(path, 'validation_data_names_pos.csv'), dtype=None).tolist()
    validation_data_names_neg_ = np.genfromtxt(os.path.join(path, 'validation_data_names_neg.csv'), dtype=None).tolist()
    validation_data_labels_pos_ = np.genfromtxt(os.path.join(path, 'validation_data_labels_pos.csv'), dtype=None).tolist()
    validation_data_labels_neg_ = np.genfromtxt(os.path.join(path, 'validation_data_labels_neg.csv'), dtype=None).tolist()

    # TODO (optional) if images need cropping, use the 'crop_images()' function above separately

    # shuffle
    train_data_ = train_data_names_pos_ + train_data_names_neg_
    train_labels_ = train_data_labels_pos_ + train_data_labels_neg_

    validation_data_ = validation_data_names_pos_ + validation_data_names_neg_
    validation_labels_ = validation_data_labels_pos_ + validation_data_labels_neg_

    everything = list(zip(train_data_, train_labels_))
    rd.shuffle(everything)
    train_data_, train_labels_ = zip(*everything)

    everything = list(zip(validation_data_, validation_labels_))
    rd.shuffle(everything)
    validation_data_, validation_labels_ = zip(*everything)

    test_images_names = 'test_images.csv'
    with open(test_images_names, 'wr') as my_file:
        for line in range(len(validation_data_)/2, len(validation_data_)):
            my_file.write(str(validation_data_[line]) + '\n')

    test_images_labels = 'test_images_labels.csv'
    with open(test_images_labels, 'wr') as my_file:
        for line in range(len(validation_labels_ )/2, len(validation_labels_)):
            my_file.write(str(validation_labels_[line]) + '\n')

    # create empty arrays
    train_data_array = np.zeros(shape=(len(train_data_), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    train_labels_array = np.zeros(shape=(len(train_labels_), pc.NUM_CLASSES))
    validation_data_array = np.zeros(shape=(len(validation_data_), pc.IMAGE_HEIGHT, pc.IMAGE_WIDTH, pc.NUM_CHANNELS))
    validation_labels_array = np.zeros(shape=(len(validation_labels_), pc.NUM_CLASSES))

    def ohe(old, new):
        for item in range(0, len(old)):
            new[item][old[item]] = 1
        return new

    for image in range(0, len(train_data_)):
        train_data_array[image] = ndimage.imread(train_data_[image])[:, :, 0:3]
    for image in range(0, len(validation_data_)):
        validation_data_array[image] = ndimage.imread(validation_data_[image])[:, :, 0:3]

    train_labels = ohe(train_labels_, train_labels_array)
    validation_labels = ohe(validation_labels_, validation_labels_array)

    # temp to test CNN
    train_labels = train_labels_
    validation_labels = validation_labels_

    return [train_data_array, train_labels, validation_data_array, validation_labels]
    pass


# load a specific dataset
def load_human_detection_data():
    # load data
    location_path = '/home/gabi/Documents/datasets/INRIAPerson'
    train_path = 'train_64x128_H96'
    validation_path = 'test_64x128_H96'

    data_list_folder = '/home/gabi/PycharmProjects/uatu/data/INRIA'

    if not os.path.exists(data_list_folder):
        print('folder does not exist, made it')
        os.makedirs(data_list_folder)
    if len(os.listdir(data_list_folder)) <= 0:
        print('folder empty, making files')
        # write list then load data
        train_data_path_pos = os.path.join(location_path, train_path, 'real_cropped_images_pos')
        train_data_names_pos = os.listdir(train_data_path_pos)
        with open(os.path.join(data_list_folder, 'train_data_names_pos.csv'), 'wr') as my_file:
            for item in train_data_names_pos:
                my_file.write(str(os.path.join(train_data_path_pos, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'train_data_labels_pos.csv'), 'wr') as my_file:
            for item in train_data_names_pos:
                my_file.write(str(1) + '\n')

        train_data_path_neg = os.path.join(location_path, train_path, 'real_cropped_images_neg')
        train_data_names_neg = os.listdir(train_data_path_neg)
        with open(os.path.join(data_list_folder, 'train_data_names_neg.csv'), 'wr') as my_file:
            for item in train_data_names_neg:
                my_file.write(str(os.path.join(train_data_path_neg, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'train_data_labels_neg.csv'), 'wr') as my_file:
            for item in train_data_names_neg:
                my_file.write(str(0) + '\n')

        validation_data_path_pos = os.path.join(location_path, validation_path, 'real_cropped_images_pos')
        validation_data_names_pos = os.listdir(validation_data_path_pos)
        with open(os.path.join(data_list_folder, 'validation_data_names_pos.csv'), 'wr') as my_file:
            for item in validation_data_names_pos:
                my_file.write(str(os.path.join(validation_data_path_pos, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'validation_data_labels_pos.csv'), 'wr') as my_file:
            for item in validation_data_names_pos:
                my_file.write(str(1) + '\n')

        validation_data_path_neg = os.path.join(location_path, validation_path, 'real_cropped_images_neg')
        validation_data_names_neg = os.listdir(validation_data_path_neg)
        with open(os.path.join(data_list_folder, 'validation_data_names_neg.csv'), 'wr') as my_file:
            for item in validation_data_names_neg:
                my_file.write(str(os.path.join(validation_data_path_neg, item)) + '\n')
        # make the labels
        with open(os.path.join(data_list_folder, 'validation_data_labels_neg.csv'), 'wr') as my_file:
            for item in validation_data_names_neg:
                my_file.write(str(0) + '\n')

    data = load_INRIA_data(data_list_folder)
    return data


def get_wrong_predictions():
    folder = 'wrong_predictions'

    paths = np.genfromtxt('test_images.csv', dtype=None).tolist()
    ans = np.genfromtxt('wrong_predictions.txt', dtype=None).tolist()

    if os.path.exists(folder):
        shutil.rmtree(folder)

    os.mkdir(folder)


    for line in range(0, len(ans)):
        step = ans[line].split(',')[1]
        if step == 'testing':
            target = ans[line].split(',')[3]
            prediction = ans[line].split(',')[5]
            if not target == prediction:
                bla = paths[line].split('/')
                thing = paths[line].split('/')[-1]
                copyfile(paths[line], os.path.join(folder, thing))

resize_image(0.5)