import csv
import os
import shutil

image_dir = "/Users/rocky/Desktop/car_train_data/image_txt/"
train_val_dir = "/Users/rocky/Desktop/car_train_data/train_val_txt/"

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

if not os.path.exists(train_val_dir):
    os.makedirs(train_val_dir)

csv_reader = csv.reader(open("/Users/rocky/Desktop/car_train_data/train.csv"))
count = -1
for line in csv_reader:
    count += 1
    if count == 0:
        continue

    with open(image_dir + line[0].split('.')[0] + ".txt", 'a+') as f:
        width = float(line[3]) - float(line[1])
        height = float(line[4]) - float(line[2])
        x_center = float(line[1]) + width / 2
        y_center = float(line[2]) + height / 2
        f.write('1' + ' ' + str(x_center / 676) + ' ' + str(y_center / 380) + ' '
                + str(width / 676) + ' ' + str(height / 380) + "\n")
        shutil.copy("/Users/rocky/Desktop/car_train_data/train_images/" + line[0], image_dir + line[0])
    if count % 10 != 0:
        with open(train_val_dir + "train.txt", "a+") as f:
            f.write(image_dir + line[0] + "\n")
    else:
        with open(train_val_dir + "val.txt", "a+") as f:
            f.write(image_dir + line[0] + "\n")





