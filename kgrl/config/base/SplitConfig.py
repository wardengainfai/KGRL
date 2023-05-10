from cuco import config_parser, Config


@config_parser(module_name = 'base.split')
class SplitConfig(Config):
    def __init__(self, train: float, test: float, valid: float):
        assert train + test + valid == 1.0, 'Train, test, valid portions must add up to 1'

        self.train = train
        self.test = test
        self.valid = valid

    @property
    def as_tuple(self):
        return (self.train, self.test, self.valid)
