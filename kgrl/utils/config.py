def push_kwargs(config: dict = None, **kwargs):
    if config is None:
        return kwargs

    result = dict(config)

    for key, value in kwargs.items():
        result[key] = value

    return result


def replace_kwargs(config: dict = None, **kwargs):
    if config is None:
        return kwargs
    return config


