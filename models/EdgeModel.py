import os, time
import torch
import torch.nn as nn
import torch.optim as optim
from .networks.EdgeModel_networks import EdgeGenerator, Discriminator
from .losses import AdversarialLoss, PerceptualLoss, StyleLoss
from utils.display import stitch_images, imsave
from utils.metrics import EdgeAccuracy
from utils.file import create_dir

class ConfigManager():
    def __init__(self):
        self.GPU = [0]              # list of gpu ids
        self.LR = 0.0001            # learning rate
        self.D2G_LR = 0.1           # discriminator/generator learning rate ratio
        self.MAX_ITERS = 2E+6       # maximum number of iterations to train the model
        self.BETA1 = 0.0            # adam optimizer beta1
        self.BETA2 = 0.9            # adam optimizer beta2
        self.GAN_LOSS = "hinge"
        self.EDGE_THRESHOLD = 0.5
        self.FM_LOSS_WEIGHT = 10
class EdgeCompleter(nn.Module):
    def __init__(self, config=None):
        super(EdgeCompleter, self).__init__()
        if config is None:
            self.config = ConfigManager()
        else:
            self.config = config

        self.cuda = "cuda" if len(self.config.GPU)>0 else "cpu"
        # generator input: [grayscale(1) + edge(1) + mask(1)]
        # discriminator input: (grayscale(1) + edge(1))
        generator = EdgeGenerator(use_spectral_norm=True).to(self.cuda)
        discriminator = Discriminator(in_channels=2, use_sigmoid=self.config.GAN_LOSS != 'hinge').to(self.cuda)
        if len(self.config.GPU) > 1:
            generator = nn.DataParallel(generator, self.config.GPU)
            discriminator = nn.DataParallel(discriminator, self.config.GPU)
        self.add_module('generator', generator)
        self.add_module('discriminator', discriminator)

        l1_loss = nn.L1Loss()
        adversarial_loss = AdversarialLoss(type=self.config.GAN_LOSS)
        self.add_module('l1_loss', l1_loss)
        self.add_module('adversarial_loss', adversarial_loss)

        #psnr = PSNR(255.0).to(self.cuda)
        #self.add_module("psnr", psnr)
        edgeacc = EdgeAccuracy(self.config.EDGE_THRESHOLD).to(self.cuda)
        self.add_module("edgeacc", edgeacc)

        self.gen_optimizer = optim.Adam(
            params=generator.parameters(),
            lr=float(self.config.LR),
            betas=(self.config.BETA1, self.config.BETA2)
        )

        self.dis_optimizer = optim.Adam(
            params=discriminator.parameters(),
            lr=float(self.config.LR) * float(self.config.D2G_LR),
            betas=(self.config.BETA1, self.config.BETA2)
        )

    def process(self, images, edges, masks):
        # zero optimizers
        self.gen_optimizer.zero_grad()
        self.dis_optimizer.zero_grad()

        # process outputs
        outputs = self(images, edges, masks)
        gen_loss = 0
        dis_loss = 0

        # discriminator loss
        dis_input_real = torch.cat((images, edges), dim=1)
        dis_input_fake = torch.cat((images, outputs.detach()), dim=1)
        dis_real, dis_real_feat = self.discriminator(dis_input_real)        # in: (grayscale(1) + edge(1))
        dis_fake, dis_fake_feat = self.discriminator(dis_input_fake)        # in: (grayscale(1) + edge(1))
        dis_real_loss = self.adversarial_loss(dis_real, True, True)
        dis_fake_loss = self.adversarial_loss(dis_fake, False, True)
        dis_loss += (dis_real_loss + dis_fake_loss) / 2

        # generator adversarial loss
        gen_input_fake = torch.cat((images, outputs), dim=1)
        gen_fake, gen_fake_feat = self.discriminator(gen_input_fake)        # in: (grayscale(1) + edge(1))
        gen_gan_loss = self.adversarial_loss(gen_fake, True, False)
        gen_loss += gen_gan_loss

        # generator feature matching loss
        gen_fm_loss = 0
        for i in range(len(dis_real_feat)):
            gen_fm_loss += self.l1_loss(gen_fake_feat[i], dis_real_feat[i].detach())
        gen_fm_loss = gen_fm_loss * self.config.FM_LOSS_WEIGHT
        gen_loss += gen_fm_loss

        return outputs, gen_loss, dis_loss

    def forward(self, images, edges, masks):
        edges_masked = (edges * (1 - masks))
        images_masked = (images * (1 - masks)) + masks
        inputs = torch.cat((images_masked, edges_masked, masks), dim=1)
        outputs = self.generator(inputs)                                    # in: [grayscale(1) + edge(1) + mask(1)]
        return outputs

    def backward(self, gen_loss=None, dis_loss=None):
        if gen_loss is not None:
            gen_loss.backward()
        self.gen_optimizer.step()

        if dis_loss is not None:
            dis_loss.backward()
        self.dis_optimizer.step()

    def train_steps(self, dataloader):
        self.train()
        metrics = {"gen_loss" : 0.0, "dis_loss" : 0.0}
        for batch, items in enumerate(dataloader, start=1):
            start = time.time()
            images, grays, edges, masks = (item.to(self.cuda) for item in items)
            # Compute prediction error
            outputs, gen_loss, dis_loss = self.process(grays, edges, masks)
            # Backpropage errors and update parameters
            self.backward(gen_loss, dis_loss)
            end = time.time()
            # Compute average metrics for this epoch.
            if batch%10==0:
                print(f"{batch} steps {end-start:.3f} secs => gen loss : {gen_loss.item():.3f}, dis_loss : {dis_loss.item():.3f}")
            metrics["gen_loss"] = metrics["gen_loss"]*(batch-1)/batch + gen_loss.item()/batch
            metrics["dis_loss"] = metrics["dis_loss"]*(batch-1)/batch + dis_loss.item()/batch
        return metrics

    def test_steps(self, dataloader, epoch):
        self.eval()
        with torch.no_grad():
            metrics = {"gen_loss" : 0.0, "dis_loss" : 0.0, "precision" : 0.0, "recall" : 0.0}
            for batch, items in enumerate(dataloader, start=1):
                images, grays, edges, masks = (item.to(self.cuda) for item in items)
                if batch == 1:
                    self.sample(images, grays, edges, masks, epoch)
                # Compute prediction error
                outputs, gen_loss, dis_loss = self.process(grays, edges, masks)
                # Compute average metrics for this epoch.
                precision, recall = self.edgeacc(edges * masks, outputs * masks)
                metrics["gen_loss"] = metrics["gen_loss"]*(batch-1)/batch + gen_loss.item()/batch
                metrics["dis_loss"] = metrics["dis_loss"]*(batch-1)/batch + dis_loss.item()/batch
                metrics["precision"] = metrics["precision"]*(batch-1)/batch + precision.item()/batch
                metrics["recall"] = metrics["recall"]*(batch-1)/batch + recall.item()/batch
        return metrics

    def sample(self, images, grays, edges, masks, epoch):
        self.eval()
        # edge model
        inputs = (grays * (1 - masks)) + masks
        edges_masked = (edges * (1 - masks))
        outputs = self(grays, edges, masks)
        outputs_merged = (outputs * masks) + (edges * (1 - masks))

        image_per_row = 2
        if images.shape[0] <= 6:
            image_per_row = 1
        images = stitch_images(
            self.postprocess(images),
            self.postprocess(inputs),
            self.postprocess(edges),
            self.postprocess(edges_masked),
            self.postprocess(outputs_merged),
            img_per_row = image_per_row
        )

        path = os.path.join(self.exp_path, "samples")
        name = os.path.join(path, str(epoch).zfill(5) + ".png")
        create_dir(path)
        print('\nsaving sample ' + name)
        images.save(name)

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()
