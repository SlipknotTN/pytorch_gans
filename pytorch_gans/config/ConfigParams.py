import configparser
from pytorch_gans.constants.Constants import Constants as Const


class ConfigParams(object):

    def __init__(self, file):

        config = configparser.ConfigParser()
        config.read_file(open(file))

        # Model
        self.zdim = config.getint(Const.ConfigSection.model, "zdim")
        self.architecture = config.get(Const.ConfigSection.model, "architecture")
        self.image_size = config.getint(Const.ConfigSection.model, "image_size", fallback=64)
        self.input_channels = config.getint(Const.ConfigSection.model, "input_channels", fallback=3)
        self.preprocess_type = config.get(Const.ConfigSection.model, "preprocess_type", fallback="pytorch_default")

        # HyperParameters
        self.epochs = config.getint(Const.ConfigSection.hyperparameters, "epochs")
        self.batch_size = config.getint(Const.ConfigSection.hyperparameters, "batch_size")
        self.learning_rate = config.getfloat(Const.ConfigSection.hyperparameters, "learning_rate")
        self.optimizer = config.get(Const.ConfigSection.hyperparameters, "optimizer")
        # if self.optimizer != "SGD":
        #     raise Exception("Only SGD optimizer supported")
        # self.momentum = config.getfloat(Const.ConfigSection.hyperparameters, "momentum")
        self.beta1 = config.getfloat(Const.ConfigSection.hyperparameters, "beta1")
