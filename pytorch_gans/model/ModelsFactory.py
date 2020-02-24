from model.GeneratorDCGAN import GeneratorDCGAN
from model.DiscriminatorDCGAN import DiscriminatorDCGAN


class ModelsFactory(object):

    @classmethod
    def create(cls, config, num_classes):

        generator = None
        discriminator = None

        if config.g_architecture == "dcgan":

            # TODO: Create Generator with upsample + conv instead of conv transpose
            generator = GeneratorDCGAN(config)

        else:

            raise Exception("Generator model architecture " + config.g_architecture + " not supported")

        if config.g_architecture == "dcgan":

            discriminator = DiscriminatorDCGAN(config)

        else:

            raise Exception("Discriminator model architecture " + config.d_architecture + " not supported")

        return generator, discriminator
