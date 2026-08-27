# torch-utils

Reliable, focused utilities for PyTorch and Lightning projects.

> [!NOTE]
> The project is in early development and does not expose stable utilities yet.

## Planned utilities

- Finite-input checks for normalization layers such as `SafeBatchNorm`.
- A Lightning `PreciseBN` callback for recomputing BatchNorm running statistics.
- A Lightning mixin that manages schedule-free AdamW train/evaluation state.

Each integration will be isolated so users only install the frameworks they need.

## License

`torch-utils` is released under the MIT License.
