from pytorch_gans.model.GeneratorCDCGAN import GeneratorCDCGAN
from pytorch_gans.model.GeneratorDCGAN import GeneratorDCGAN
from pytorch_gans.model.GeneratorDCGAN_Upsample import GeneratorDCGAN_Upsample
from pytorch_gans.model.DiscriminatorDCGAN import DiscriminatorDCGAN
from pytorch_gans.model.DiscriminatorCDCGAN import DiscriminatorCDCGAN


class ModelsFactory(object):

    @classmethod
    def create(cls, config, num_classes):

        if config.g_architecture == "dcgan":

            generator = GeneratorDCGAN(config)

        elif config.g_architecture == "dcgan_upsample":

            generator = GeneratorDCGAN_Upsample(config)

        elif config.g_architecture == "cdcgan":

            generator = GeneratorCDCGAN(config, num_classes)

        else:

            raise Exception("Generator model architecture " + config.g_architecture + " not supported")

        if config.d_architecture == "dcgan":

            discriminator = DiscriminatorDCGAN(config)

        elif config.d_architecture == "cdcgan":

            discriminator = DiscriminatorCDCGAN(config, num_classes)

        else:

            raise Exception("Discriminator model architecture " + config.d_architecture + " not supported")

        return generator, discriminator
