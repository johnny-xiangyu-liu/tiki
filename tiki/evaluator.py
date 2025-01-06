from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tiki.train_config import TrainConfig
from tiki.train_state import TrainState


class Evaluator:
    def __init__(self,
                 train_config: TrainConfig,
                 loss_fn: Callable[[
                     Any,  # model output
                     Any,  # target
                 ], torch.tensor],
                 dataloader: DataLoader,
                 log_writer: SummaryWriter
                 ):
        self.config = train_config
        self.loss_fn = loss_fn
        self.dataset = dataloader.dataset
        self.dataloader = dataloader
        self.writer = log_writer

    def to_device(self, param):
        return param.to(self.config.device)

    def eval_batch(self, model, x, target):
        output = model(x)
        loss = self.loss_fn(output, target)
        batch_loss = loss.detach()
        return output, batch_loss

    def eval(self, model, train_state: TrainState):
        assert self.dataset.type in ['eval', 'val', 'validation']

        model.eval()
        with torch.no_grad():
            self.to_device(model)

            print("----- Start Eval Epoch ----- at state:", train_state)
            total_loss = 0
            total_N = 0
            total_diff = 0
            for idx, (x, target) in enumerate(self.dataloader):
                x = self.to_device(x)
                target = self.to_device(target)

                total_N += target.shape[0]
                output, batch_loss = self.eval_batch(model, x, target)
                total_loss += batch_loss.item()

                prediction = torch.argmax(output, dim=1)
                diff = (prediction != target)
                total_diff += torch.sum(diff)

            total_avg_loss = total_loss / total_N

            print('step', train_state.step)
            if self.writer is not None:
                self.writer.add_scalar("Eval Loss", total_avg_loss,
                                       global_step=train_state.step)

                self.writer.add_scalar("Eval Prediction Diff",
                                       total_diff,
                                       global_step=train_state.step)
            print(f"----- Final. Eval Loss (avg) {total_avg_loss}, diff: {total_diff} ------: ")
