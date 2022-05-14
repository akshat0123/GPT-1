from typing import Dict

from torch import set_grad_enabled, argmax, Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.utils import RollingCounter


class Trainer:


    def __init__(self, model: 'Model', crit: 'Loss', opt: 'Optimizer', 
                 sch: 'Scheduler', device: str):
        """ Initialize trainer

        Args:
            model: model to train
            crit: loss function to train with
            opt: optimizer to train with
            sch: learning rate scheduler
            device: device to place trainer on
        """

        self.device = device
        self.model = model
        self.crit = crit
        self.opt = opt
        self.sch = sch
        

    def run_epoch(self, loader: DataLoader, train_mode: bool=True) -> Dict[str, int]:
        """ Run a single epoch of training

        Args:
            loader: data loader to train / evaluate data on
            train_mode: flag indicating whether epoch is training or evaluation

        Returns:
            (Dict[str, int]): dictionary containing epoch metrics
        """

        loss_metric, err_metric = RollingCounter(1000), RollingCounter(1000)
        progress = tqdm(total=len(loader), desc='LR: | Loss: | Err: ')

        self.model.train(mode=train_mode)
        with set_grad_enabled(train_mode):
            for x, y, ignore in loader:
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                ignore = ignore.to(device=self.device)
                loss, err = self.step(x, y, ignore, train_mode)
                loss_metric.add(loss)
                err_metric.add(err)

                progress.set_description(
                    f'LR: {self.sch.get_last_lr()[-1]:.8f} | '
                    f'Loss: {loss_metric.rolling_average():.8f} | '
                    f'Err: {err_metric.rolling_average():.8f}'
                )
                progress.update(1)

        return {
            'total_average_loss': loss_metric.total_average(),
            'rolling_average_loss': loss_metric.rolling_average(),
            'total_average_err': err_metric.total_average(),
            'rolling_average_err': err_metric.rolling_average(),
        }


    def step(self, x: Tensor, y: Tensor, ignore: Tensor, 
             train_mode: bool=True) -> (float, float):
        """ Run one training step

        Args:
            x: input
            y: labels
            ignore: input indices to ignore
            train_mode: flag indicating whether to train model during step

        Returns:
            (float): loss
            (float): error
        """

        if train_mode:
            self.model.zero_grad()
            self.opt.zero_grad()

        y_pred = self.model(x, ignore)
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y = y.view(-1)
        loss = self.crit(y_pred, y)

        if train_mode:
            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.sch.step()

        y_pred = argmax(y_pred, dim=1)
        err = (y_pred!=y).sum() / y.shape[0]

        return loss.item(), err


