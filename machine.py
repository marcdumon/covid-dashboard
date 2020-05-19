# --------------------------------------------------------------------------------------------------------
# 2019/12/26
# src - machine.py
# md
# --------------------------------------------------------------------------------------------------------
from distutils.dir_util import remove_tree, copy_tree
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import torchvision as thv
from ignite.contrib.handlers import CosineAnnealingScheduler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.handlers import EarlyStopping
from ignite.metrics import Accuracy, Loss, Precision, Recall, TopKCategoricalAccuracy, ConfusionMatrix
from my_tools.confusion_matrix import pretty_plot_confusion_matrix
from my_tools.delayed_lr_scheduler import DelayedCosineAnnealingLR, DelayerScheduler
from my_tools.lr_finder import LRFinder
from my_tools.make_graphviz_graph import make_dot
from my_tools.python_tools import print_file, now_str
from my_tools.pytorch_tools import DeNormalize, summary
from skimage import io
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision.models import vgg16, resnet18
from torchvision.transforms import transforms

from configuration import rcp, cfg
from models.standard_models import MNSIT_Simple, imagenette2_Simple


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # self.cnn = MNSIT_Simple()
        # self.cnn = imagenette2_Simple()
        # self.vgg = vgg16()
        self.cnn = resnet18(pretrained=False)
        self.cnn.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)
        # x = self.vgg(x)

        return x


def run_training(model, train, valid, optimizer, loss, lr_find=False):
    print_file(f'Experiment: {rcp.experiment}\nDescription:{rcp.description}', f'{rcp.base_path}description.txt')
    print_file(model, f'{rcp.models_path}model.txt')
    print_file(get_transforms(), f'{rcp.models_path}transform_{rcp.stage}.txt')
    # Data
    train.transform = get_transforms()
    valid.transform = get_transforms()
    train.save_csv(f'{rcp.base_path}train_df_{rcp.stage}.csv')
    valid.save_csv(f'{rcp.base_path}valid_df_{rcp.stage}.csv')
    train_loader = DataLoader(train, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)
    valid_loader = DataLoader(valid, batch_size=rcp.bs, num_workers=8, shuffle=rcp.shuffle_batch)

    if lr_find: lr_finder(model, optimizer, loss, train_loader, valid_loader)

    one_batch = next(iter(train_loader))
    dot = make_dot(model(one_batch[0].to(cfg.device)), params=dict(model.named_parameters()))
    dot.render(f'{rcp.models_path}graph', './', format='png', cleanup=True)
    summary(model, one_batch[0].shape[-3:], batch_size=rcp.bs, device=cfg.device, to_file=f'{rcp.models_path}summary_{rcp.stage}.txt')

    # Engines
    trainer = create_supervised_trainer(model, optimizer, loss, device=cfg.device)
    t_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                              'nll': Loss(loss),
                                                              'precision': Precision(average=True),
                                                              'recall': Recall(average=True),
                                                              'topK': TopKCategoricalAccuracy()
                                                              }, device=cfg.device)
    v_evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                              'nll': Loss(loss),
                                                              'precision_avg': Precision(average=True),
                                                              'recall_avg': Recall(average=True),
                                                              'topK': TopKCategoricalAccuracy(),
                                                              'conf_mat': ConfusionMatrix(num_classes=len(valid.classes), average=None),
                                                              }, device=cfg.device)

    # Tensorboard
    tb_logger = TensorboardLogger(log_dir=f'{rcp.tb_log_path}{rcp.stage}')
    tb_writer = tb_logger.writer
    tb_logger.attach(trainer, log_handler=OptimizerParamsHandler(optimizer, "lr"), event_name=Events.EPOCH_STARTED)
    tb_logger.attach(trainer, log_handler=WeightsHistHandler(model), event_name=Events.EPOCH_COMPLETED)
    tb_logger.attach(trainer, log_handler=WeightsScalarHandler(model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=GradsScalarHandler(model), event_name=Events.ITERATION_COMPLETED)
    tb_logger.attach(trainer, log_handler=GradsHistHandler(model), event_name=Events.EPOCH_COMPLETED)

    @trainer.on(Events.EPOCH_COMPLETED)
    def tb_and_log_training_stats(engine):
        t_evaluator.run(train_loader)
        v_evaluator.run(valid_loader)
        tb_and_log_train_valid_stats(engine, t_evaluator, v_evaluator, tb_writer)

    @trainer.on(Events.ITERATION_COMPLETED(every=int(1 + len(train_loader) / 100)))
    def print_dash(engine):
        print('-', sep='', end='', flush=True)

    if cfg.show_batch_images:
        @trainer.on(Events.STARTED)
        def show_batch_images(engine):
            imgs, lbls = next(iter(train_loader))
            denormalize = DeNormalize(**rcp.transforms.normalize)
            for i in range(len(imgs)):
                imgs[i] = denormalize(imgs[i])
            imgs = imgs.to(cfg.device)
            grid = thv.utils.make_grid(imgs)
            tb_writer.add_image('images', grid, 0)
            tb_writer.add_graph(model, imgs)
            tb_writer.flush()

    if cfg.show_top_losses:
        @trainer.on(Events.COMPLETED)
        def show_top_losses(engine, k=6):
            nll_loss = nn.NLLLoss(reduction='none')
            df = predict_dataset(model, valid, nll_loss, transform=None, bs=rcp.bs, device=cfg.device)
            df.sort_values('loss', ascending=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            for i, row in df.iterrows():
                img = cv2.imread(str(row['fname']))
                img = th.as_tensor(img.transpose(2, 0, 1))  # #CHW
                tag = f'TopLoss_{engine.state.epoch}/{row.loss:.4f}/{row.target}/{row.pred}/{row.pred2}'
                tb_writer.add_image(tag, img, 0)
                if i >= k - 1: break
            tb_writer.flush()

    if cfg.tb_projector:
        images, labels = train.select_n_random(250)
        # get the class labels for each image
        class_labels = [train.classes[lab] for lab in labels]
        # log embeddings
        features = images.view(-1, images.shape[-1] * images.shape[-2])
        tb_writer.add_embedding(features, metadata=class_labels, label_img=images)

    if cfg.log_pr_curve:
        @trainer.on(Events.COMPLETED)
        def log_pr_curve(engine):
            """
            1. gets the probability predictions in a test_size x num_classes Tensor
            2. gets the preds in a test_size Tensor
            takes ~10 seconds to run
            """
            class_probs = []
            class_preds = []
            with th.no_grad():
                for data in valid_loader:
                    imgs, lbls = data
                    imgs, lbls = imgs.to(cfg.device), lbls.to(cfg.device)
                    output = model(imgs)
                    class_probs_batch = [th.softmax(el, dim=0) for el in output]
                    _, class_preds_batch = th.max(output, 1)
                    class_probs.append(class_probs_batch)
                    class_preds.append(class_preds_batch)
            test_probs = th.cat([th.stack(batch) for batch in class_probs])
            test_preds = th.cat(class_preds)

            for i in range(len(valid.classes)):
                """ Takes in a "class_index" from 0 to 9 and plots the corresponding precision-recall curve"""
                tensorboard_preds = test_preds == i
                tensorboard_probs = test_probs[:, i]

                tb_writer.add_pr_curve(f'{rcp.stage}/{valid.classes[i]}',
                                       tensorboard_preds,
                                       tensorboard_probs,
                                       global_step=engine.state.epoch,
                                       num_thresholds=127)
                tb_writer.flush()

    print()

    if cfg.lr_scheduler:
        # lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=.5, min_lr=1e-7, verbose=True)
        # v_evaluator.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step(v_evaluator.state.metrics['nll']))
        lr_scheduler = DelayedCosineAnnealingLR(optimizer, 10, 5)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lambda engine: lr_scheduler.step(trainer.state.epoch))

    if cfg.early_stopping:
        def score_function(engine):
            score = -1 * round(engine.state.metrics['nll'], 5)
            # score = engine.state.metrics['accuracy']
            return score

        es_handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
        v_evaluator.add_event_handler(Events.COMPLETED, es_handler)

    if cfg.save_last_checkpoint:
        @trainer.on(Events.EPOCH_COMPLETED(every=1))
        def save_last_checkpoint(engine):
            checkpoint = {}
            objects = {'model': model, 'optimizer': optimizer}
            if cfg.lr_scheduler: objects['lr_scheduler'] = lr_scheduler
            for k, obj in objects.items():
                checkpoint[k] = obj.state_dict()
            th.save(checkpoint, f'{rcp.models_path}last_{rcp.stage}_checkpoint.pth')

    if cfg.save_best_checkpoint:
        def score_function(engine):
            score = -1 * round(engine.state.metrics['nll'], 5)
            # score = engine.state.metrics['accuracy']
            return score

        objects = {'model': model, 'optimizer': optimizer}
        if cfg.lr_scheduler: objects['lr_scheduler'] = lr_scheduler

        save_best = Checkpoint(objects, DiskSaver(f'{rcp.models_path}', require_empty=False, create_dir=True),
                               n_saved=4, filename_prefix=f'best_{rcp.stage}',
                               score_function=score_function, score_name='val_loss',
                               global_step_transform=global_step_from_engine(trainer))
        v_evaluator.add_event_handler(Events.EPOCH_COMPLETED(every=1), save_best)
        load_checkpoint = False

        if load_checkpoint:
            resume_epoch = 6
            cp = f'{rcp.models_path}last_{rcp.stage}_checkpoint.pth'
            obj = th.load(f'{cp}')
            Checkpoint.load_objects(objects, obj)

            @trainer.on(Events.STARTED)
            def resume_training(engine):
                engine.state.iteration = (resume_epoch - 1) * len(engine.state.dataloader)
                engine.state.epoch = resume_epoch - 1

    if cfg.save_confusion_matrix:
        @trainer.on(Events.STARTED)
        def init_best_loss(engine):
            engine.state.metrics['best_loss'] = 1e99

        @trainer.on(Events.EPOCH_COMPLETED)
        def confusion_matric(engine):
            if engine.state.metrics['best_loss'] > v_evaluator.state.metrics['nll']:
                engine.state.metrics['best_loss'] = v_evaluator.state.metrics['nll']
                cm = v_evaluator.state.metrics['conf_mat']
                cm_df = pd.DataFrame(cm.numpy(), index=valid.classes, columns=valid.classes)
                pretty_plot_confusion_matrix(cm_df, f'{rcp.results_path}cm_{rcp.stage}_{trainer.state.epoch}.png', False)

    if cfg.log_stats:
        class Hook:
            def __init__(self, module):
                self.name = module[0]
                self.hook = module[1].register_forward_hook(self.hook_fn)
                self.stats_mean = 0
                self.stats_std = 0

            def hook_fn(self, module, input, output):
                self.stats_mean = output.mean()
                self.stats_std = output.std()

            def close(self):
                self.hook.remove()

        hookF = [Hook(layer) for layer in list(model.cnn.named_children())]

        @trainer.on(Events.ITERATION_COMPLETED)
        def log_stats(engine):
            std = {}
            mean = {}
            for hook in hookF:
                tb_writer.add_scalar(f'std/{hook.name}', hook.stats_std, engine.state.iteration)
                tb_writer.add_scalar(f'mean/{hook.name}', hook.stats_mean, engine.state.iteration)

    cfg.save_yaml()
    rcp.save_yaml()
    print(f'# batches: train: {len(train_loader)}, valid: {len(valid_loader)}')
    trainer.run(data=train_loader, max_epochs=rcp.max_epochs)
    tb_writer.close()
    tb_logger.close()
    return model


def get_transforms():
    tsfm = []
    if rcp.transforms.topilimage: tsfm += [transforms.ToPILImage()]
    if rcp.transforms.randomrotation: tsfm += [transforms.RandomRotation(rcp.transforms.randomrotation)]
    if rcp.transforms.randomverticalflip: tsfm += [transforms.RandomVerticalFlip(rcp.transforms.randomverticalflip)]
    if rcp.transforms.randomhorizontalflip: tsfm += [transforms.RandomHorizontalFlip(rcp.transforms.randomhorizontalflip)]
    if rcp.transforms.colorjitter: tsfm += [transforms.ColorJitter(**rcp.transforms.colorjitter)]

    # if rcp.transforms.randomcrop: tsfm += [transforms.RandomCrop(rcp.transforms.randomcrop)]
    if rcp.transforms.resize: tsfm += [transforms.Resize((rcp.transforms.resize, rcp.transforms.resize))]
    if rcp.transforms.totensor: tsfm += [transforms.ToTensor()]
    if rcp.transforms.normalize: tsfm += [transforms.Normalize(**rcp.transforms.normalize)]

    return transforms.Compose(tsfm)


def setup_experiment():
    """
    Create directories for experiment:
        ../temp_reports
            /experiment
                /yyyymmdd_hhmmss
                    /models
                    /results
                    /src

    """
    # Create paths
    Path(f'{rcp.models_path}').mkdir(parents=True, exist_ok=True)
    Path(f'{rcp.results_path}').mkdir(parents=True, exist_ok=True)
    # Copy src
    source = '../src'
    destination = f'{rcp.src_path}'
    copy_tree(source, destination)


def close_experiment(experiment: str, datetime: str):
    """
    move experiment to ../reports
    move ../tensorboard/experiment to ../reports
    """
    source = f'{cfg.temp_report_path}{experiment}/{datetime}/'
    tb_source = f'../tensorboard/{experiment}/{datetime}/'
    destination = f'../experiments/{experiment}/{datetime}/'
    copy_tree(source, destination, verbose=2)
    copy_tree(tb_source, f'{destination}tensorboard', verbose=2)
    remove_tree(source, verbose=2)
    remove_tree(tb_source, verbose=2)
    exit()


def lr_finder(model, optimizer, loss, train_loader, valid_loader=None, device=cfg.device):
    lr_find = LRFinder(model, optimizer, loss, device)
    lr_find.range_test(train_loader=train_loader, val_loader=valid_loader, end_lr=1e-2, num_iter=100)
    lr_find.plot()
    lr_find.reset()
    exit()


def tb_and_log_train_valid_stats(engine, t_evaluator, v_evaluator, tb_writer):
    # t_evaluator.run(train_loader)
    t_metrics = t_evaluator.state.metrics
    t_avg_acc = t_metrics['accuracy']
    t_avg_nll = t_metrics['nll']
    t_avg_prec = t_metrics['precision']
    t_avg_rec = t_metrics['recall']
    t_topk = t_metrics['topK']

    # v_evaluator.run(valid_loader)
    v_metrics = v_evaluator.state.metrics
    v_avg_acc = v_metrics['accuracy']
    v_avg_nll = v_metrics['nll']
    v_avg_prec = v_metrics['precision_avg']
    v_avg_rec = v_metrics['recall_avg']
    v_topk = v_metrics['topK']

    print()
    print_file(f'{now_str("mm-dd hh:mm:ss")} |'
               f'Ep:{engine.state.epoch:3} | '
               f'acc: {t_avg_acc:.5f}/{v_avg_acc:.5f} | '
               f'loss: {t_avg_nll:.5f}/{v_avg_nll:.5f} | '
               f'prec: {t_avg_prec:.5f}/{v_avg_prec:.5f} | '
               f'rec: {t_avg_rec:.5f}/{v_avg_rec:.5f} |'
               f'topK: {t_topk:.5f}/{v_topk:.5f} |',
               f'{rcp.results_path}train_log_{rcp.stage}.txt')

    tb_writer.add_scalar("0_train/acc", t_avg_acc, engine.state.epoch)
    tb_writer.add_scalar("0_train/loss", t_avg_nll, engine.state.epoch)
    tb_writer.add_scalar("0_train/prec", t_avg_prec, engine.state.epoch)
    tb_writer.add_scalar("0_train/rec", t_avg_rec, engine.state.epoch)
    tb_writer.add_scalar("0_train/topK", t_topk, engine.state.epoch)

    tb_writer.add_scalar("0_valid/acc", v_avg_acc, engine.state.epoch)
    tb_writer.add_scalar("0_valid/loss", v_avg_nll, engine.state.epoch)
    tb_writer.add_scalar("0_valid/prec", v_avg_prec, engine.state.epoch)
    tb_writer.add_scalar("0_valid/rec", v_avg_rec, engine.state.epoch)
    tb_writer.add_scalar("0_valid/topK", v_topk, engine.state.epoch)
    tb_writer.flush()


def predict_dataset(model, dataset, loss_fn, bs, transform=None, device=cfg.device):
    """
    Takes a model, dataset and loss_fn returns a dataframe with columns = [fname, targets, loss, pred]
    """
    if transform:
        dataset.transform = transform
    else:
        dataset.transform = get_transforms()
    dataloader = DataLoader(dataset, bs, shuffle=False, num_workers=8)
    df = pd.DataFrame()
    df['fname'] = dataset.data
    df['target'] = dataset.targets
    model.to(device)
    loss = []
    pred = []
    pred2 = []
    target = []
    for images, targets in dataloader:
        model.eval()
        with th.no_grad():
            images, targets = images.to(device), targets.to(device)
            logits = model(images)
            l = loss_fn(logits, targets)
            p = th.argmax(logits, dim=1)
            # 2nd argmax
            p2 = th.topk(logits, 2, dim=1)  # returns namedtuple (values, indices)
            p2 = p2.indices[:, 1]  # second column
            loss += list(l.to('cpu').numpy())
            pred += list(p.to('cpu').numpy())
            pred2 += list(p2.to('cpu').numpy())
            target += list(targets.to('cpu').numpy())
    df['loss'] = loss
    df['pred'] = pred
    df['pred2'] = pred2
    df['target'] = target

    return df


if __name__ == '__main__':
    # close_experiment('baseline', '20200103_230746')
    pass
