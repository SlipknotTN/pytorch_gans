from model.GeneratorDCGAN import GeneratorDCGAN
from model.DiscriminatorDCGAN import DiscriminatorDCGAN


class ModelsFactory(object):

    @classmethod
    def create(cls, config, num_classes):

        if config.architecture == "dcgan":

            return GeneratorDCGAN(config), DiscriminatorDCGAN(config)

        # elif config.architecture == "cdcgan":
        #
        #     return GeneratorCDCGAN(num_classes), DiscriminatorCDCGAN(num_classes)

        else:

            raise Exception("Model architecture " + config.architecture + " not supported")