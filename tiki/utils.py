import os
from os import listdir
from os.path import isfile, join
from pathlib import Path

from tiki.train_config import TrainConfig
from tiki.train_state import TrainState


def find_model_checkpoint_path(train_config, train_state: TrainState = None):
    if train_state is None:
        root_path = train_config.model_dir_path()
        if os.path.isdir(root_path):
            all_checkpoints = sorted(all_files(root_path, name_only=True))
            # return the last file in all the checkpoints
            step = int(all_checkpoints[-1]) if all_checkpoints else None
            train_state = TrainState(step=step) if step is not None else None

    if train_state is None:
        return None
    return model_checkpoint_path(train_config, train_state)


def model_checkpoint_path(train_config: TrainConfig, train_state: TrainState):
    step = train_state.step
    return train_config.model_dir_path().joinpath(f"{step:010}")


def traced_pytorch_model_path(train_config: TrainConfig, train_state: TrainState):
    step = train_state.step
    return train_config.traced_model_dir_path().joinpath(f"traced_{step:010}.pt")


def coreml_model_path(train_config: TrainConfig, train_state: TrainState):
    step = train_state.step
    return train_config.coreml_model_dir_path().joinpath(f"coreml_model_{step:010}.mlpackage")


def unify_channels(a_tensor):
    def grayscale_to_rgb(tensor):
        # Check if the image is already in 3 channels
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)

        return tensor

    def remove_alpha_channel(tensor):
        # Check if the image is already in 3 channels
        if tensor.shape[0] == 4:
            # Drop the alpha channel (assuming it's the last channel)
            return tensor[:3, :, :]

        return tensor

    return grayscale_to_rgb(remove_alpha_channel(a_tensor))


# List all the visible files under a path.
def all_files(path: Path, name_only=False) -> list[str]:
    if name_only:
        return [f for f in listdir(path) if isfile(join(path, f)) and not f.startswith('.')]
    return [str(path.joinpath(f)) for f in listdir(path) if isfile(join(path, f)) and not f.startswith('.')]
