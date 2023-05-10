from dataclasses import dataclass

from cuco import config_parser, Config

from .FrequencyUnit import FrequencyUnit


@config_parser(module_name = 'base.frequency')
@dataclass
class Frequency(Config):
    value: int
    unit: FrequencyUnit

    @property
    def as_tuple(self):
        return (self.value, self.unit.value)

    @staticmethod
    def load(value: int, unit: str):
        return Frequency(value = value, unit = FrequencyUnit(unit))
