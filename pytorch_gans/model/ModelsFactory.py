from pytorch_gans.model.GeneratorDCGAN import GeneratorDCGAN
from pytorch_gans.model.DiscriminatorDCGAN import DiscriminatorDCGAN
from pytorch_gans.model.GeneratorDCGAN_Upsample import GeneratorDCGAN_Upsample


class ModelsFactory(object):

    @classmethod
    def create(cls, config, num_classes):

        generator = None
        discriminator = None

        if config.g_architecture == "dcgan":

            generator = GeneratorDCGAN(config)

        elif config.g_architecture == "dcgan_upsample":

            generator = GeneratorDCGAN_Upsample(config)

        else:

            raise Exception("Generator model architecture " + config.g_architecture + " not supported")

        if config.d_architecture == "dcgan":

            discriminator = DiscriminatorDCGAN(config)

        else:

            raise Exception("Discriminator model architecture " + config.d_architecture + " not supported")

        return generator, discriminator
