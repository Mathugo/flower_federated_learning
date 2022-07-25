import glob, os, shutil
import random
import numpy
output_dir="clients"
client_name="client"
nb_clients = 3
valid_freq = 0.2

img_freq = [0]*nb_clients    
img_freq_train_valid = [{'train':0,'valid':0} for k in range(nb_clients)]

dir = "data"



def split_into_n_clients(nb_clients: int):
    
    for file in glob.glob(dir+"/*/*.jpg", recursive=True):
        #print(file)
        dir_label = file.split(os.path.basename(file))[0]
        label = dir_label.split(dir)[1].replace('/', '')

        client_i = random.randint(0, nb_clients-1)
        #client_i = random.randint(0, nb_clients-1)

        client_folder = f"{output_dir}/{client_name}{client_i}"
        label_folder = f"{client_folder}/{label}"
        img_freq[client_i]+=1

        train = f"{client_folder}/train/"
        valid = f"{client_folder}/valid/"

        if not os.path.isdir(client_folder):
            os.mkdir(client_folder)

        if not os.path.isdir(train):
            os.mkdir(train)
        if not os.path.isdir(valid):
            os.mkdir(valid)

        train_folder = os.path.join(train, os.path.basename(label_folder))
        if not os.path.isdir(train_folder):
            os.mkdir(train_folder)

        valid_folder = os.path.join(valid, os.path.basename(label_folder))
        if not os.path.isdir(valid_folder):
            os.mkdir(valid_folder)
        
        nb = random.uniform(1, 10)
        if nb <= valid_freq*10:
            # valid
            to_copy = valid_folder
            img_freq_train_valid[client_i]["valid"]+=1
        else:
            to_copy = train_folder
            img_freq_train_valid[client_i]["train"]+=1

        shutil.copy(file, to_copy)
        print(to_copy)
    
    print(f"Number of images in clients {img_freq}")
    print(f"Number of train valid in clients {img_freq_train_valid}")

    """
    print(f"Frequency {img_freq}")
    # split into valid
    for client in glob.glob(output_dir+"/*"):
        train = f"{client}/train/"
        valid = f"{client}/valid/"

        if not os.path.isdir(train):
            os.mkdir(train)
        if not os.path.isdir(valid):
            os.mkdir(valid)
        
        print(client)
        
        for files in glob.glob(client+"/*", recursive=True):
            print(files)
        
    """

split_into_n_clients(3)

    
