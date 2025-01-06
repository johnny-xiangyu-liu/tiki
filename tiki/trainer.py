#  Copyright (c) [2025] [Xiangyu Liu]
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.


from copy import deepcopy
from typing import Any, Callable

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tiki.evaluator import Evaluator
from tiki.train_config import TrainConfig
from tiki.train_state import TrainState


class Trainer:
    def __init__(self,
                 train_config: TrainConfig,
                 model: torch.nn.Module,
                 optimizer,
                 loss_fn: Callable[[Any,  # model
                                    Any,  # model output
                                    Any,  # target
                                    ], torch.tensor],
                 dataloader: DataLoader,
                 checkpoint_manager,
                 evaluator: "Evaluator",
                 log_writer: SummaryWriter,
                 start_state=None,
                 ):
        self.config = deepcopy(train_config)
        if start_state is None:
            self.state = TrainState()
        else:
            self.state = deepcopy(start_state)

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataset = dataloader.dataset
        self.dataloader = dataloader
        self.checkpoint_manager = checkpoint_manager
        self.evaluator = evaluator
        self.log_writer = log_writer

    def to_device(self, param):
        return param.to(self.config.device)

    def should_log(self):
        print_every = self.config.print_every
        if print_every is None or print_every <= 0:
            return False
        if print_every == 1:
            return True

        return self.state.step % print_every == 0

    def do_log(self, target, output, batch_loss) -> None:
        print("Training State:", self.state)
        if self.log_writer is None:
            return
        prediction = torch.argmax(output, dim=1)
        diff = torch.sum(prediction != target)

        self.log_writer.add_scalar("Train Batch Loss", batch_loss,
                                   self.state.step)
        self.log_writer.add_scalar("Train Batch Diff", diff,
                                   self.state.step)

    def should_checkpoint(self):
        checkpoint_every = self.config.checkpoint_every
        if checkpoint_every is None:
            return False

        if checkpoint_every == 1:
            return True

        step = self.state.step
        return step % checkpoint_every == 0

    def do_checkpoint(self):
        if self.evaluator is not None:
            self.evaluator.eval(self.model, self.state)
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.store(self.state, self.model, self.optimizer, self.dataloader)

        # reset the model to train
        self.model.train()

    def train_batch(self, x, target):
        optimizer = self.optimizer
        # Zero your gradients for every batch!
        optimizer.zero_grad()
        output = self.model(x)
        loss = self.loss_fn(self.model, output, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        optimizer.step()

        batch_loss = loss.detach()

        return output, batch_loss

    def train_epoch(self):
        data_size = len(self.dataloader)
        start_index = self.state.step % data_size

        for idx, (x, target) in enumerate(self.dataloader):
            if idx < start_index:
                # TODO this is inefficient, as it loads the data and ignores them.
                continue
            checkpointed = False
            self.state.step_start()

            x = self.to_device(x)
            target = self.to_device(target)

            output, batch_loss = self.train_batch(x, target)

            self.state.step_update(batch_loss.item())

            if self.should_checkpoint():
                self.do_checkpoint()
                checkpointed = True

            if self.should_log():
                self.do_log(target, output, batch_loss)

            self.state.step_end()
            if self.state.step >= self.config.max_steps:
                if not checkpointed:
                    self.do_checkpoint()
                break

    def train(self):
        assert self.dataset.type == 'train'
        print("------- Training Start -------")
        self.model.train()
        self.to_device(self.model)

        # checkpoint the initial state.
        if self.state.step == 0:
            self.do_checkpoint()
        while self.state.step < self.config.max_steps:
            self.state.epoch_start()
            self.train_epoch()
            self.state.epoch_end()
        print("------- Training Finished -------")
