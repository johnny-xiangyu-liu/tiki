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


from dataclasses import dataclass, fields
from pathlib import Path

import torch

from tiki.state_dictable_dataclass import StateDictableDataclass


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = torch.device('cpu')

    return device


@dataclass(init=False)
class TrainConfig(StateDictableDataclass):
    # the root dir of the output
    output_dir_str: str
    # arbitrary name of the train. (aka id)
    name: str
    batch_size: int
    lr: float
    print_every: int = 100
    checkpoint_every: int = 100
    max_steps: int = 1000
    shuffle: bool = False
    device = get_device()

    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

    def output_dir(self):
        return Path(self.output_dir_str).joinpath(self.name)

    def model_dir_path(self):
        return self.output_dir().joinpath('model')

    def traced_model_dir_path(self):
        return self.output_dir().joinpath('traced_model')

    def coreml_model_dir_path(self):
        return self.output_dir().joinpath('coreml')

    def tensorboard_path(self):
        return self.output_dir().joinpath('tensor_board')
