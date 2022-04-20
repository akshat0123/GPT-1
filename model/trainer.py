
from torch import set_grad_enabled, no_grad, argmax
from tqdm import tqdm

from model.utils import RollingCounter


class Trainer:


    def __init__(self, model, crit, opt, sch, device):
        self.device = device
        self.model = model
        self.crit = crit
        self.opt = opt
        self.sch = sch
        

    def run_epoch(self, loader, train=True):

        barsize = len(loader.dataset)
        progress = tqdm(total=barsize, desc='LR: | Loss: | Err: ')

        loss_metric, err_metric = RollingCounter(1000), RollingCounter(1000)
        self.model.train(mode=train)

        with set_grad_enabled(train):
            for x, y, padding, line_idx in loader:
                padding = padding.to(device=self.device)
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                loss, err = self.step(x, y, padding, train)
                loss_metric.add(loss)
                err_metric.add(err)

                if line_idx > progress.n:
                    progress.set_description(
                        f'LR: {self.sch.get_last_lr()[-1]:.8f} | '
                        f'Loss: {loss_metric.rolling_average():.8f} | '
                        f'Err: {err_metric.rolling_average():.8f}'
                    )
                    progress.update(line_idx - progress.n)

            progress.update(barsize - progress.n)

        return {
            'total_average_loss': loss_metric.total_average(),
            'rolling_average_loss': loss_metric.rolling_average(),
            'total_average_err': err_metric.total_average(),
            'rolling_average_err': err_metric.rolling_average(),
        }


    def step(self, x, y, padding, train=True):

        if train:
            self.model.zero_grad()
            self.opt.zero_grad()

        y_pred = self.model(x, padding)
        y_pred = y_pred.view(-1, y_pred.size(-1))
        y = y.view(-1)
        loss = self.crit(y_pred, y)

        if train:
            loss.backward()
            self.opt.step()
            self.sch.step()

        y_pred = argmax(y_pred, dim=1)
        err = (y_pred!=y).sum() / y.shape[0]

        return loss.item(), err


