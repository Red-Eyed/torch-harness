"""Tests for saving recursive PyTorch model structures to text files."""

from pathlib import Path
from typing import cast

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from torch_utils.lightning import FileModelStructure


class _StructureModel(LightningModule):
    """Provide a nested network with informative PyTorch representations."""

    def __init__(self) -> None:
        """Create a small nested network for exact representation checks."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    @override
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Run the nested network during the integration fit."""
        inputs = cast(torch.Tensor, args[0] if args else kwargs["inputs"])
        return self.network(inputs)

    @override
    def training_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
        """Return a scalar loss for the one-step integration fit."""
        batch = cast(tuple[torch.Tensor], args[0] if args else kwargs["batch"])
        return self(batch[0]).square().mean()

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Construct the optimizer required by Lightning's fit loop."""
        return torch.optim.SGD(self.parameters(), lr=0.1)


def test_writes_exact_recursive_model_representation(tmp_path: Path) -> None:
    """Persist the same full structure produced by printing the model."""
    output_path = tmp_path / "artifacts" / "model-structure.txt"
    output_path.parent.mkdir()
    output_path.write_text("stale structure", encoding="utf-8")
    model = _StructureModel()
    callback = FileModelStructure(output_path=output_path)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[callback],
    )
    dataloader = DataLoader(
        TensorDataset(torch.ones(1, 4)),
        batch_size=1,
    )

    trainer.fit(model, train_dataloaders=dataloader)

    assert output_path.read_text(encoding="utf-8") == f"{model}\n"


def test_non_global_process_does_not_write(tmp_path: Path) -> None:
    """Leave structure persistence to Lightning's global-zero process."""

    class _NonGlobalTrainer:
        """Provide the distributed state used by the callback hook."""

        is_global_zero = False

    output_path = tmp_path / "model-structure.txt"
    trainer = cast(Trainer, _NonGlobalTrainer())
    callback = FileModelStructure(output_path=output_path)

    callback.on_fit_start(trainer, _StructureModel())

    assert not output_path.exists()
