from math import floor


def stringify_timediff(timediff: int):
    n_minutes_ = floor(timediff) // 60
    n_seconds = timediff - n_minutes_ * 60
    n_hours_ = n_minutes_ // 60
    n_minutes = n_minutes_ - n_hours_ * 60
    n_days = n_hours_ // 24
    n_hours = n_hours_ - n_days * 24

    if n_days > 0:
        return f'{n_days}d {n_hours}h {n_minutes}m {n_seconds:.3f}s'
    if n_hours > 0:
        return f'{n_hours}h {n_minutes}m {n_seconds:.3f}s'
    if n_minutes > 0:
        return f'{n_minutes}m {n_seconds:.3f}s'
    return f'{n_seconds:.3f}s'
