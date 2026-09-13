"""Reference execution, outside measured build and runtime stages."""

from collections.abc import Sequence

import torch
from torch import Tensor

from tobench.workloads import BaseWorkload


class TorchEagerReference:
    @torch.inference_mode()
    def run(self, workload: BaseWorkload, inputs: Sequence[Tensor]) -> Tensor:
        output = workload.eval()(*inputs)
        if not isinstance(output, Tensor):
            raise TypeError("reference must return a Tensor")
        # Snapshot outputs which might alias input or module storage.
        output = output.detach().clone()
        if output.is_cuda:
            torch.cuda.synchronize(output.device)
        return output
