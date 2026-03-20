from verl.workers.rollout import HFRollout as _HFRollout


class HFRolloutShim(_HFRollout):
    """
    Compatibility shim for newer vendored verl versions.

    MMSearch-R1's FSDP worker builds the HF rollout with the historical
    `(module, config)` constructor, while newer `BaseRollout` requires
    `(config, model_config, device_mesh)` during initialization. The upstream
    HF rollout implementation still relies only on `self.module` and
    `self.config`, so we bypass the new base initializer and provide the
    missing attributes directly.
    """

    def __init__(self, module, config, model_config=None, device_mesh=None):
        self.module = module
        self.config = config
        self.model_config = model_config
        self.device_mesh = device_mesh

    async def update_weights(self, *args, **kwargs):
        return None

    async def resume(self, *args, **kwargs):
        return None

    async def release(self, *args, **kwargs):
        return None
