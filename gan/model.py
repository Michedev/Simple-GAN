import pytorch_lightning as pl
import torch
import torchvision.utils


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
        pos_map_expanded = self.pos_map.expand(x.size(0), -1, -1, -1)
        x = torch.cat([x, pos_map_expanded], dim=1)
        return self.linear_mapping(x)


class GAN(pl.LightningModule):

    def __init__(self, latent_size: int, discriminator_steps: int,
                 generator: torch.nn.Module, discriminator: torch.nn.Module,
                 discriminator_loss_log_steps: int, generator_loss_log_steps: int,
                 train_log_images_steps: int, log_norm_steps: int):
        super().__init__()
        self.log_norm_steps = log_norm_steps
        self.latent_size = latent_size
        self.latent_dim = latent_size
        self.discriminator_steps = discriminator_steps
        self.generator = generator
        self.discriminator = discriminator
        self.it_discriminator = 0
        self.iteration = 0
        self.iteration_discriminator = 0
        self.it_generator = 0
        self.discriminator_loss_log_steps = discriminator_loss_log_steps
        self.generator_loss_log_steps = generator_loss_log_steps
        self.train_log_images_steps = train_log_images_steps
        self.automatic_optimization = False
        self.it_phase = 0

    def forward(self, batch_size=None):
        batch_size = batch_size or 1
        z = torch.randn(batch_size, self.latent_size)
        return self.generator(z)

    def training_step(self, batch, batch_idx):
        opt_gen, opt_discr = self.optimizers()
        X, y = batch
        with torch.no_grad():
            X = (X * 2) - 1  # now in [-1, 1]
        fake_batch = self(X.size(0))
        if self.iteration % self.train_log_images_steps == 0:
            self._log_stats_gen_images(fake_batch, is_train=True)
        if self.it_phase < self.discriminator_steps:
            opt_discr.zero_grad()
            fake_batch = fake_batch.detach()
            p_true = self.discriminator(X) + 1e-5
            p_fake = self.discriminator(fake_batch) + 1e-5

            loss = - p_true.log() - (1 - p_fake).log()
            loss = loss.mean(dim=0).sum()
            if self.it_discriminator % self.discriminator_loss_log_steps == 0:
                self.log('train/loss_discriminator', loss, prog_bar=True)
            self.it_discriminator += 1
            loss.backward()
            opt_discr.step()
            self.it_phase += 1
        else:
            opt_gen.zero_grad()
            p_fake = self.discriminator(fake_batch) + 1e-5
            loss = - p_fake.log().mean(dim=0).sum()
            if self.it_generator % self.generator_loss_log_steps == 0:
                self.log('train/loss_generator', loss, prog_bar=True)
            self.it_generator += 1
            opt_gen.step()
            self.it_phase = 0
        if self.iteration % self.log_norm_steps == 0:
            self._log_grad_norm()
        self.iteration += 1
        return loss

    def validation_step(self, batch, batch_idx):
        X, y = batch
        fake_batch = self(X.size(0))
        self._log_stats_gen_images(fake_batch, is_train=False)
        p_true = self.discriminator(X)
        p_fake = self.discriminator(fake_batch)
        loss_discriminator = p_true.log() + (1 - p_fake).log()
        loss_discriminator = - loss_discriminator.mean(dim=0).sum()
        self.log('val/loss_discriminator', loss_discriminator)
        loss_generator = (1 - p_fake).log().mean(dim=0).sum()
        self.log('val/loss_generator', loss_generator)
        self.iteration += 1
        # return dict(loss_generator=loss_generator, loss_discriminator=loss_discriminator, fake_batch=fake_batch)

    @torch.no_grad()
    def _log_stats_gen_images(self, fake_batch, is_train: bool):
        is_train = 'train' if is_train else 'val'
        grid_fake_batch = torchvision.utils.make_grid(
            fake_batch
        )
        self.logger.experiment.add_image(f'{is_train}/gen_images', grid_fake_batch, global_step=self.iteration)
        for stat in ['min', 'max', 'mean', 'std']:
            self.log(f'{is_train}/fake_img_{stat}', getattr(fake_batch[0], stat)())

    @torch.no_grad()
    def _log_grad_norm(self):
        norm_d = sum(
            [torch.norm(p.grad) for p in self.discriminator.parameters() if hasattr(p, 'grad') and p.grad is not None])
        norm_g = sum(
            [torch.norm(p.grad) for p in self.generator.parameters() if hasattr(p, 'grad') and p.grad is not None])
        self.log('grad/norm_discriminator', norm_d)
        self.log('grad/norm_generator', norm_d)
        self.log('grad/norm_model', norm_d + norm_g)


    def configure_optimizers(self):
        return [torch.optim.Adam(self.generator.parameters(), 1e-3),
                torch.optim.Adam(self.discriminator.parameters(), 1e-3)]