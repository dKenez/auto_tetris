## TO BE TUNED
# batch_size: 20, 40, 60
# lr: 1e-4, 1e-3, 1e-2, 1e-1, 
# layers: 3, 4, 5
# optimizer: ADAM, RMS, SGD
# loss: DiceBCE, Dice, IoU

## DON'T TUNE
# epochs: 50, 100
# image_size: 128


import toml
from pathlib import Path
from loss import DiceBCELoss, DiceLoss, IoULoss

class Hyperparams:
    def __init__(self, path: Path):
        toml_dict = toml.load(path)

        # To tune
        self.batch_size = toml_dict['batch_size']
        self.epochs = toml_dict['epochs']
        self.lr = toml_dict['lr']
        self.layers = toml_dict['layers']
        self.optimizer = toml_dict['optimizer']
        self.loss = toml_dict['loss']

        # Not to tune
        self.image_size = toml_dict['image_size']

    def model_name(self):
        return f"roof_surface_model_B{self.batch_size}_E{self.epochs}_lr{self.layers}_L{self.layers}_{self.optimizer}_{self.loss}.pth"

    # using property decorator
    # a getter function
    @property
    def loss_fn(self):
        if self.loss == "DiceBCE":
            return DiceBCELoss()
        if self.loss == "Dice":
            return DiceLoss()
        if self.loss == "IoU":
            return IoULoss()

    # # using property decorator
    # # a getter function
    # @property
    # def loss_fn(self):
    #     if self.loss == "DiceBCE":
    #         return DiceBCELoss()
    #     if self.loss == "Dice":
    #         return DiceLoss()
    #     if self.loss == "IoU":
    # return Adam


if __name__ == "__main__":
    base_path = Path(__file__).parent.parent
    
    h = Hyperparams(base_path / 'train_conf.toml')

    print(h.model_name())