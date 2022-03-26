from typing import Dict, List

from tqdm import tqdm
import torch

from model.utils import RollingCounter


class Trainer:


    def __init__(self, model: torch.nn.Module, optimizer: torch.optim, 
                 loss_fn: torch.nn, scheduler: torch.optim.lr_scheduler):
        """ Initialize trainer module

        Args:
            model: model to be trained
            optimizer: optimizer for learning
            loss_fn: loss function to optimize
            scheduler: learning rate scheduler
        """
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.model = model
        self.metrics = {}


    def reset_metrics(self) -> None:
        """ Zero out internal metrics
        """

        self.error = RollingCounter(1000)
        self.loss = RollingCounter(1000)


    def train(self, loader: torch.utils.data.DataLoader) -> (float, float):
        """ Train on batches in loader

        Args:
            loader: data loader containing training data in batches

        Returns:
            (float): average loss
            (float): average error
        """

        barsize = len(loader.dataset)
        progress = tqdm(total=barsize, desc='Train Loss: | Error: | LR: ')
        self.reset_metrics()
        self.model.train()

        for batch, line_idx in loader:
            batch = batch.to(self.model.device)
            self.optimizer.zero_grad()

            loss = self.step(batch)
            loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            desc = (f'Train | Rolling Loss: {self.loss.rolling_average():.10f} '
                    f'| Rolling Error: {self.error.rolling_average():.10f} '
                    f'| LR: {self.scheduler.get_last_lr()[0]:.10f}')
            progress.set_description(desc)

            if line_idx > progress.n:
                progress.update(line_idx - progress.n)

        progress.update(progress.total - progress.n)
        
        self.metrics.update({
            'train_rolling_error': self.error.rolling_average(),
            'train_rolling_loss': self.loss.rolling_average(),
            'train_total_error': self.error.total_average(),
            'train_total_loss': self.loss.total_average()
        })


    def validate(self, loader: torch.utils.data.DataLoader) -> (float, float):
        """ Validate using batches in loader

        Args:
            loader: data loader containing validation data in batches

        Returns:
            (float): average loss
            (float): average error
        """

        barsize = len(loader.dataset)
        progress = tqdm(total=barsize, desc='Val Loss: | Error: ')
        self.reset_metrics()
        self.model.eval()

        with torch.no_grad():
            for batch, line_idx in loader:
                batch = batch.to(self.model.device)
                loss = self.step(batch)

                desc = (f'Train | Total Loss: {self.loss.total_average():.10f} '
                        f'| Total Error: {self.error.total_average():.10f}')
                progress.set_description(desc)

                if line_idx > progress.n:
                    progress.update(line_idx - progress.n)

        progress.update(progress.total - progress.n)

        self.metrics.update({
            'dev_total_error': self.error.total_average(),
            'dev_total_loss': self.loss.total_average()
        })


    def step(self, batch: torch.Tensor):
        """ Run training / validation step on provided batch

        Args:
            batch: tensor of byte pair ids to train model on 
        """
        x, y = batch[:, :-1, :], batch[:, -1, :]
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)

        y, yhat = torch.argmax(y, dim=1), torch.argmax(yhat, dim=1)
        avg_batch_err = torch.mean((yhat != y).float()).item()
        avg_batch_loss = loss.item()

        self.error.add(avg_batch_err)
        self.loss.add(avg_batch_loss)

        return loss


    def get_checkpoint(self) -> Dict[str, str]:
        """  Get dictionary of state dictionaries for checkpointing

        Returns:
            (Dict[str, str]): dictionary mapping trainer components to their
                              state dictionaries for checkpointing
        """

        return {
            'train_rolling_error': self.metrics['train_rolling_error'],
            'train_rolling_loss': self.metrics['train_rolling_loss'],
            'train_total_error': self.metrics['train_total_error'],
            'train_total_loss': self.metrics['train_total_loss'],
            'dev_total_error': self.metrics['dev_total_error'],
            'dev_total_loss': self.metrics['dev_total_loss'],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'model': self.model.state_dict()
        }
