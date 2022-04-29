
from torch import set_grad_enabled, argmax
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from model.utils import RollingCounter


class Trainer:


    def __init__(self, model, crit, opt, sch, device):
        self.device = device
        self.model = model
        self.crit = crit
        self.opt = opt
        self.sch = sch
        

    def run_epoch(self, loader, train_mode=True):

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


    def step(self, x, y, ignore, train_mode=True):

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


