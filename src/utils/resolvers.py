r"""Custom functions to register by Hydra"""
from typing import Callable, Dict

from omegaconf import OmegaConf


def register_custom_resolvers(extra_resolvers: Dict[str, Callable] = None):
    """Wrap your main function with this.
    You can pass extra kwargs, e.g. `version_base` introduced in 1.2.
    """
    extra_resolvers = extra_resolvers or {}
    for name, resolver in extra_resolvers.items():
        OmegaConf.register_new_resolver(name, resolver)


def register_resolvers():
    register_custom_resolvers({
        "eval": eval,
        "len": len
    })
