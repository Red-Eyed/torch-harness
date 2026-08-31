"""Persist PyTorch module structures as readable text files."""

from pathlib import Path

from lightning.pytorch import Callback, LightningModule, Trainer
from loguru import logger
from typing_extensions import override

_LOGGER = logger.bind(component="FileModelStructure")


class FileModelStructure(Callback):
    """Write a Lightning module's recursive PyTorch structure to a file.

    The generated UTF-8 text is equivalent to ``print(pl_module)``. It shows
    registered submodules and their constructor details without executing a
    forward pass or requiring example inputs. Only the global-zero process writes
    the file. Parent directories are created automatically, and an existing file
    is replaced.

    This is a module-structure tree, not a traced computational graph: runtime
    tensor operations that are not registered modules do not appear.

    Args:
        output_path: Destination for the recursive model representation.

    Raises:
        OSError: If the destination directory or structure file cannot be written.

    Example:
        Save the complete model structure when fitting starts::

            >>> from pathlib import Path
            >>> from lightning.pytorch import Trainer
            >>> from torch_harness.lightning import FileModelStructure
            >>> structure = FileModelStructure(
            ...     output_path=Path("artifacts/model-structure.txt"),
            ... )
            >>> trainer = Trainer(callbacks=[structure])
    """

    def __init__(self, *, output_path: Path) -> None:
        """Store the destination used when fitting starts."""
        self._output_path = output_path

    @override
    def on_fit_start(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
    ) -> None:
        """Persist the module structure from the global-zero process."""
        if not trainer.is_global_zero:
            return

        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        self._output_path.write_text(f"{pl_module}\n", encoding="utf-8")
        _LOGGER.info("Saved model structure to {}", self._output_path)
