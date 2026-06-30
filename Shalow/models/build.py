"""Local Unet2D factory for Shalow experiments."""

from __future__ import annotations

from typing import Literal

from src.train.fix_window_train import TrainConfig

GeneratorName = Literal["gUnet", "shalow_gUnet", "shalow_gUnet_xs"]

_GENERATORS: dict[str, str] = {
    "gUnet": "gUnet",
    "shalow_gUnet": "Shalow_gUnet",
    "shalow_gUnet_xs": "Shalow_gUnet_xs",
}


def import_unet2d(generator: str = "gUnet"):
    """Return Unet2D class from the selected Shalow generator module."""
    module_name = _GENERATORS.get(generator)
    if module_name is None:
        allowed = ", ".join(sorted(_GENERATORS))
        raise ValueError(f"Unknown generator {generator!r}; choose one of: {allowed}")
    if module_name == "gUnet":
        from gUnet import Unet2D
    elif module_name == "Shalow_gUnet":
        from Shalow_gUnet import Unet2D
    else:
        from Shalow_gUnet_xs import Unet2D
    return Unet2D


def build_model(crop_h: int, crop_w: int, config: TrainConfig):
    generator = getattr(config, "generator", "gUnet")
    Unet2D = import_unet2d(generator)
    par = {
        "lb": 3,
        "nf": config.iD,
        "nx": crop_h,
        "ny": crop_w,
        "inp_shift": 0.0,
        "inp_scale": 1.0,
        "out_shift": 0.0,
        "out_scale": 1.0,
        "t_shift": 0.0,
        "t_scale": 1.0,
    }
    return Unet2D(
        dim=config.unet_dim,
        Par=par,
        dim_mults=config.unet_dim_mults,
        channels=par["lb"] * par["nf"],
        **(
            {"dropout": float(getattr(config, "unet_dropout", 0.0))}
            if generator == "gUnet"
            else {}
        ),
    )
