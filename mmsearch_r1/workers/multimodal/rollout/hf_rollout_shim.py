from verl.workers.rollout import HFRollout as _HFRollout


class HFRolloutShim(_HFRollout):
    """
    Compatibility shim for verl versions where HFRollout is abstract.
    Implement required abstract methods as no-ops.
    """

    def update_weights(self, *args, **kwargs):
        return None

    def resume(self, *args, **kwargs):
        return None

    def release(self, *args, **kwargs):
        return None
