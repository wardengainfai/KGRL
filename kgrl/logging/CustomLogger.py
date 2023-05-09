import datetime
import os
import tempfile
from typing import List, Optional

from stable_baselines3.common.logger import Logger, INFO, make_output_format, KVWriter


class CustomLogger(Logger):

    def __init__(self, folder: Optional[str], output_formats: List[KVWriter], level: int = INFO):
        super().__init__(folder, output_formats)
        self.level = level

    def dump(self, step: int = 0):
        print(
            f'n episodes = {self.name_to_value["time/episodes"]:-5d} \t' +
            f'mean episode length = {self.name_to_value["rollout/ep_len_mean"]:-10.3f} \t' +
            f'mean reward = {self.name_to_value["rollout/ep_rew_mean"]:-10.3f} \t' +
            f'total timesteps = {self.name_to_value["time/total_timesteps"]:8d}'
        )


def configure(
        folder: Optional[str] = None,
        format_strings: Optional[List[str]] = None,
        level = INFO,
        init_logger: callable = None
) -> CustomLogger:
    """
    Configure the current logger.

    :param folder: the save location
        (if None, $SB3_LOGDIR, if still None, tempdir/SB3-[date & time])
    :param format_strings: the output logging format
        (if None, $SB3_LOG_FORMAT, if still None, ['stdout', 'log', 'csv'])
    :return: The logger object.
    """
    if folder is None:
        folder = os.getenv("SB3_LOGDIR")
    if folder is None:
        folder = os.path.join(tempfile.gettempdir(), datetime.datetime.now().strftime("SB3-%Y-%m-%d-%H-%M-%S-%f"))
    assert isinstance(folder, str)
    os.makedirs(folder, exist_ok=True)

    log_suffix = ""
    if format_strings is None:
        format_strings = os.getenv("SB3_LOG_FORMAT", "stdout,log,csv").split(",")

    format_strings = list(filter(None, format_strings))
    output_formats = [make_output_format(f, folder, log_suffix) for f in format_strings]

    # if init_logger is not None:
    #     assert isinstance(init_logger, CustomLogger), "The init Logger should be of class CustomLogger."

    logger = (CustomLogger if init_logger is None else init_logger)(folder=folder, output_formats=output_formats, level = level)
    # Only print when some files will be saved
    if len(format_strings) > 0 and format_strings != ["stdout"]:
        logger.log(f"Logging to {folder}")
    return logger
