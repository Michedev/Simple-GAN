import pytorch_lightning as pl
import torch


class ExpandLatentSpace(torch.nn.Module):

    def __init__(self, width: int, height: int):
        super().__init__()
        self.width = width
        self.height = height

    def forward(self, z):
        z: torch.Tensor = z.view(z.size(0), z.size(1), 1, 1)
        z = z.expand(-1, -1, self.width, self.height)
        return z


class PositionalEmbedding(torch.nn.Module):

    def __init__(self, width: int, height: int, channels: int):
        super().__init__()
        self.width = width
        self.height = height
        self.channels = channels
        self.pos_map = torch.linspace(-1, 1, self.width * self.height, dtype=torch.float).reshape(1, 1, self.width, self.height)
        self.linear_mapping = torch.nn.Conv2d(channels + 1, channels, kernel_size=1)

    def forward(self, x):
        self.pos_map = self.pos_map.to(x.device)
        x = torch.cat([x, self.pos_map], dim=1)
        return self.linear_mapping(x)


class GAN(pl.LightningModule):

    def __init__(self, latent_size: int, discriminator_steps: int,
                 generator: torch.nn.Module, discriminator: torch.nn.Module,
                 discriminator_loss_log_steps: int, generator_loss_log_steps: int):
        super().__init__()
        self.latent_size = latent_size
        self.discriminator_steps = discriminator_steps
        self.generator = generator
        self.discriminator = discriminator
        self.it_discriminator = 0
        self.iteration = 0
        self.iteration_discriminator = 0
        self.iteration_generator = 0
        self.discriminator_loss_log_steps = discriminator_loss_log_steps
        self.generator_loss_log_steps = generator_loss_log_steps

    def forward(self, batch_size=None):
        batch_size = batch_size or 1
        z = torch.randn(batch_size, self.latent_size)
        return self.discriminator(z)

    def training_step(self, batch, batch_idx):
        fake_batch = self(batch.size(0))
        if self.it_discriminator < self.discriminator_steps:
            logit_true = self.discriminator(batch)
            logit_fake = self.discriminator(fake_batch)
            loss = logit_true.sigmoid().log() + (1 - logit_fake).log()
            loss = - loss.mean(dim=0).sum()
            self.log('loss/train_discriminator', loss)
            self.iteration_discriminator += 1
        else:
            with torch.no_grad():
                logit_fake = self.discriminator(fake_batch)
            self.it_discriminator = 0
            loss = (1 - logit_fake).log().mean(dim=0).sum()
            self.log('loss/train_generator', loss)
            self.iteration_generator += 1
        self.iteration += 1
        return loss
