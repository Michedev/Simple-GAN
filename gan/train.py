import hydra
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torch.utils.data import DataLoader
from gan.paths import CONFIG


@hydra.main(CONFIG, 'train')
def train(config):
    model = hydra.utils.instantiate(config.model)
    train_set = hydra.utils.instantiate(config.dataset.train)
    test_set = hydra.utils.instantiate(config.dataset.test)

    train_dl = DataLoader(train_set)
    test_dl = DataLoader(test_set)

    checkpoint = ModelCheckpoint('./', monitor='loss/train_generator', save_top_k=3)

    trainer = Trainer(callbacks=[checkpoint, ModelSummary()])

    trainer.fit(model, train_dl, test_dl)

