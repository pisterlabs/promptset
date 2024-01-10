def normalize(loss):
    """
    work over standard mean, to avoid unecessary chaos in policy, source from OpenAI
    .. avoid using pgd_loss.mean() and use pgd_loss.sum() instead
        + may stabilize learning
        - well, it normalize our advantage ( value not anymore holds advantage, +/- can be swapped )
    .. i prefer to tune { learning rates / batch size / step per learning } instead
    """
    normalize = lambda a: (a - a.mean()) / a.std()
    return normalize(loss)
