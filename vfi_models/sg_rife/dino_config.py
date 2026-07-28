import os
from typing import List
from dataclasses import dataclass, field

# Base directory for resolving relative paths (sg-rife root)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass
class DinoConfig:
    """
    Configuration for DINO integration settings.

    Attr:
        dino_loss_weight (float): Dino loss hyperparameter
        compressor_rank (int): FAPM hyperparameter
        interaction_indices (int): idx of DINO hidden states for FAPM
    """

    dino_repo_dir: str = os.path.join(_BASE_DIR, "dinov3_repo")
    dino_model_name: str = "dinov3_vits16"
    dino_checkpoint_path: str = os.path.join(
        os.path.dirname(_BASE_DIR), os.pardir, "checkpoints", "sg_rife",
        "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
    )

    dino_loss_weight: float = 0.5
    dcn_loss_weight: float = 0.0001
    l1_loss_weight: float = 0.1
    tea_loss_weight: float = 0.1
    offset_lr_mult: float = 0.25
    compressor_rank: int = 256
    interaction_indices: List[int] = field(default_factory=lambda: [8, 11])


def main():
    print(DinoConfig().dino_loss_weight)
    print(DinoConfig().dcn_loss_weight)
    print(DinoConfig().offset_lr_mult)


if __name__ == "__main__":
    main()