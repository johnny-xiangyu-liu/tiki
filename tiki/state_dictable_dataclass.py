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
