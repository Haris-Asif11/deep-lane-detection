# run_predict.py
import os
import torch
from torch.utils.data import DataLoader

from data.image_dataset import ImageDataset
from model import Model
from network.unet import UNet
from utils.experiment import Experiment

def main():
    # Open Experiment
    exp = Experiment(name='Adam_a0.0001_LW180_epochs30_AugCustom7', new=False)
    cfg_path = exp.params['cfg_path']

    # Init & load model
    model = Model(exp, net=UNet)
    model.load_trained_model()

    # Dataset / loader
    predict_dataset = ImageDataset(
        mode='Predict', dataset_name='TestSet10', size=10, seed=49, cfg_path=cfg_path
    )
    predict_loader = DataLoader(dataset=predict_dataset, batch_size=1, shuffle=False, num_workers=2)

    # Where to save predictions
    out_dir = os.path.join("data/output_data", exp.exp_name,  "prediction_results")
    os.makedirs(out_dir, exist_ok=True)

    # Run predictions
    idx_offset = 0
    for batch_idx, (images, _) in enumerate(predict_loader):
        model.predict(
            images,
            video_writer=None,
            output_dir=out_dir,
            show=True,           # set False to disable window
            idx_offset=idx_offset
        )
        idx_offset += images.size(0)

    print(f"\n✅ Saved predictions to: {os.path.abspath(out_dir)}")

if __name__ == "__main__":
    main()