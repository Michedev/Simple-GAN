import hydra
import omegaconf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torch.utils.data import DataLoader
from gan.paths import CONFIG

omegaconf.OmegaConf.register_new_resolver('sum', lambda *x: sum(float(el) for el in x))


@hydra.main(CONFIG, 'train')
def train(config):
    with open('config.yaml', 'w') as f:
        omegaconf.OmegaConf.save(config, f)
    model = hydra.utils.instantiate(config.model)
    train_set = hydra.utils.instantiate(config.dataset.train)
    test_set = hydra.utils.instantiate(config.dataset.test)

    train_dl = DataLoader(train_set, shuffle=True, batch_size=config.batch_size)
    test_dl = DataLoader(test_set, shuffle=True, batch_size=config.batch_size)

    checkpoint = ModelCheckpoint('./', filename='gan-mnist-epoch{epoch:02d}-val_loss_gen{val/loss_generator:.2f}'
                                                '-val_loss_discr{val/loss_discriminator:.2f}',
                                 save_top_k=3, auto_insert_metric_name=False, save_on_train_epoch_end=True)

    trainer = Trainer(callbacks=[checkpoint, ModelSummary(max_depth=2)], val_check_interval=config.val_interval,
                      limit_val_batches=config.val_steps, accelerator=config.accelerator, devices=config.devices)

    trainer.fit(model, train_dl, test_dl)


if __name__ == '__main__':
    train()