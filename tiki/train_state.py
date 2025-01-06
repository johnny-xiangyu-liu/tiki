#  Copyright (c) 2025 [Xiangyu Liu]
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

import time
from dataclasses import dataclass, fields
from typing import Union

import torch

from tiki.state_dictable_dataclass import StateDictableDataclass


@dataclass(init=False)
class TrainState(StateDictableDataclass):
    epoch_idx: int = 0
    step: int = 0
    # how long it takes to process 1 step of training
    avg_step_time: float = 0
    batch_loss: Union[float, torch.Tensor] = 0
    epoch_loss: Union[float, torch.Tensor] = 0

    step_start_time = None

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

    def epoch_start(self):
        pass

    def epoch_end(self):
        self.epoch_loss = 0
        self.batch_loss = 0
        self.epoch_idx += 1

    def step_start(self):
        self.step_start_time = time.time()
        # increment the step at the start time.
        self.step += 1

    def step_update(self, batch_loss):
        self.batch_loss = batch_loss
        self.epoch_loss += batch_loss

    def step_end(self):
        step_time = time.time() - self.step_start_time
        self.step_start_time = None

        prev_step_time = self.avg_step_time * (self.step - 1)
        self.avg_step_time = (prev_step_time + step_time) / self.step
