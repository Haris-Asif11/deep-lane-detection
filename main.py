import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.transforms import Compose

from data.image_dataset import ImageDataset
from model import Model, get_device
from network.unet import UNet
from utils.experiment import Experiment
from utils.augmentation import VerticalFlip, Rotate, ColRec, GaussianBlur, GaussianNoise, ZoomIn, ZoomOut

# Creating an experiment for training.
#exp = Experiment(name='Adam_w500', new=True, overwrite=True)
exp = Experiment(name='Adam_a0.0001_LW180_epochs30_AugCustom8', new=True, overwrite=True)
cfg_path=exp.params['cfg_path']

augmentation_list = Compose([
                                  VerticalFlip(probability=0.5),
                                  Rotate(probability=0.5),
                                  ColRec(probability=0.2),
                                  GaussianNoise(probability=0.1),
                                  GaussianBlur(probability=0.2),
                                  ZoomIn(probability=0.5),
                                 ])

#augmentation_list = None


train_dataset = ConcatDataset([
        ImageDataset(dataset_name='SimSet1',size=250,cfg_path=cfg_path,augmentation=augmentation_list),
        #ImageDataset(dataset_name='SimSet2',size=250,cfg_path=cfg_path,augmentation=augmentation_list),
    ])


train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=4,
                                               shuffle=True, num_workers=0)#original batch = 4

# Test Set
test_dataset =  ImageDataset(dataset_name='Sim_Lane_Test',size=20,cfg_path=cfg_path,mode='Test', seed=5)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=1,
                                              shuffle=False, num_workers=4)

# Initialize the model
model = Model(exp,net=UNet)

#Defime optimiser parameters
optimiser= torch.optim.Adam  #original Adam
optimiser_params = {'lr': 0.0001} #original 0.0001

# Weighted Loss function
no_lane_weight = 1

lane_weight = 180  #original 500; lets try [500, 1000, 2000, 5000, 15000]

device = get_device()

# Non-lane is class 0 and lane is class 1
# loss function weight array -> [class_0_wt, class_1_wt]

weight = torch.FloatTensor([no_lane_weight, lane_weight]).to(device)
loss_func_params ={'weight' : weight}

model.setup_model(optimiser=optimiser,
                  optimiser_params=optimiser_params,
                  loss_function=torch.nn.CrossEntropyLoss,
                  loss_func_params=loss_func_params)

total_epochs = 30  # original epochs = 30

print("\t\t Avg.Loss \t Accuracy \t Lane F1 Score")
model.reset_train_stats()
for i in range(total_epochs):

    print("Epoch [{}/{}]".format(i + 1, total_epochs))

    model.net.train()
    # Train step
    for images, labels in train_loader:
        model.train_step(images, labels)
    model.print_stats(mode="Train")
    model.save_model(i)

    # Validation step
    model.net.eval()
    for image, labels in test_loader:
        model.validation_step(image, labels)
    model.print_stats(mode="Test")

model.get_best_model()
exp.del_unused_models()