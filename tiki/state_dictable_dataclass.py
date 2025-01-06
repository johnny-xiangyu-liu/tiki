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


from abc import ABC

from dataclasses import dataclass, asdict, fields


@dataclass(init=False)
class StateDictableDataclass(ABC):

    def state_dict(self):
        return asdict(self)

    def load_state_dict(self, state_dict):
        if state_dict is None:
            return
        names = set([f.name for f in fields(self)])
        for k, v in state_dict.items():
            if k in names:
                setattr(self, k, v)
