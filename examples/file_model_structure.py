"""Generate a recursive model-structure file with ``FileModelStructure``."""

from pathlib import Path
from typing import cast

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from typing_extensions import override

from torch_harness.lightning import FileModelStructure


class ExampleModel(LightningModule):
    """Provide a nested network whose structure is easy to inspect."""

    def __init__(self) -> None:
        """Create the network printed in the generated structure file."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    @override
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        """Run the nested network during the demonstration fit."""
        inputs = cast(torch.Tensor, args[0] if args else kwargs["inputs"])
        return self.network(inputs)

    @override
    def training_step(self, *args: object, **kwargs: object) -> STEP_OUTPUT:
        """Return a scalar loss for the one-step demonstration fit."""
        batch = cast(tuple[torch.Tensor], args[0] if args else kwargs["batch"])
        return self(batch[0]).square().mean()

    @override
    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Construct the optimizer required by Lightning's fit loop."""
        return torch.optim.SGD(self.parameters(), lr=0.1)


def main() -> None:
    """Fit one batch and save the model's complete recursive structure."""
    output_path = Path(__file__).with_name("model-structure.txt")
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[FileModelStructure(output_path=output_path)],
    )
    dataloader = DataLoader(
        TensorDataset(torch.ones(1, 4)),
        batch_size=1,
    )

    trainer.fit(ExampleModel(), train_dataloaders=dataloader)


if __name__ == "__main__":
    main()
