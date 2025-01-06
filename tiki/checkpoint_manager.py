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

import os
from collections import defaultdict

import torch

from tiki import utils
from tiki.train_config import TrainConfig
from tiki.train_state import TrainState


class CheckpointManager:
    def __init__(self, train_config: TrainConfig):
        self.train_config = train_config

    def store(self, train_state: TrainState, model, optimizer=None, dataloader=None, debug=False):
        path = utils.find_model_checkpoint_path(self.train_config, train_state)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        state_map = {
            'train_config_state_dict': self.train_config.state_dict(),
            'train_state_state_dict': train_state.state_dict(),
            'model_state_dict': model.state_dict(),
        }

        if optimizer is not None:
            state_map['optimizer_state_dict'] = optimizer.state_dict()

        if dataloader is not None:
            state_dict_op = getattr(dataloader, "state_dict", None)
            if callable(state_dict_op):
                state_map['dataloader_state_dict'] = state_dict_op(dataloader)
        if not debug:
            torch.save(state_map, path)
        print("Saving checkpoint to ", path)

    # Always loads onto CPU device.
    @classmethod
    def load(cls, train_config: TrainConfig, train_state: TrainState = None):
        checkpoint_path = utils.find_model_checkpoint_path(train_config, train_state)
        if checkpoint_path is not None:
            state_map = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
        else:
            state_map = {}

        state_map = defaultdict(lambda: None, state_map)
        train_config_state_dict = state_map['train_config_state_dict']
        train_state_state_dict = state_map['train_state_state_dict']
        model_state_dict = state_map['model_state_dict']
        optimizer_state_dict = state_map['optimizer_state_dict']
        dataloader_state_dict = state_map['dataloader_state_dict']

        return train_config_state_dict, train_state_state_dict, model_state_dict, optimizer_state_dict, dataloader_state_dict
