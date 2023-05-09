from .KGWrapperConfig import KGWrapperConfig

from .KGModelCheckpointConfig import KGModelCheckpointConfig


class KGUseCaseWrapperConfig(KGWrapperConfig):

    @classmethod
    def load(cls, **kwargs):
        if (checkpoint := kwargs.get('checkpoint')) is None:
            kwargs['checkpoint'] = KGModelCheckpointConfig(label = cls.label)
        else:
            checkpoint.label = cls.label

        return KGUseCaseWrapperConfig(
            **KGWrapperConfig.load(**kwargs).__dict__
        )
