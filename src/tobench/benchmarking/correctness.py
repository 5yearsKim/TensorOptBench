"""Tensor correctness checks and FP64 error statistics."""

import torch
from torch import Tensor

from tobench.core.config import CorrectnessConfig
from tobench.core.result import CorrectnessResult


class CorrectnessChecker:
    """Require matching shapes/dtypes and reject every NaN or infinity.

    Comparisons and statistics use FP64 on CPU. Absolute error statistics cover
    finite pairs only; nonfinite values always count as failures. Shape mismatch
    leaves elementwise statistics unset because elements cannot be paired.
    """

    def __init__(self, config: CorrectnessConfig | None = None) -> None:
        self.config = config or CorrectnessConfig()

    def compare(self, actual: Tensor, reference: Tensor) -> CorrectnessResult:
        if not isinstance(actual, Tensor) or not isinstance(reference, Tensor):
            raise TypeError("correctness comparison requires Tensor outputs")
        config = self.config.resolved(str(reference.dtype).removeprefix("torch."))
        common = {
            "reference": config.reference,
            "rtol": config.rtol,
            "atol": config.atol,
        }
        if not config.enabled:
            return CorrectnessResult(
                **common, status="skipped", message="Disabled by configuration"
            )
        shape_match = actual.shape == reference.shape
        dtype_match = actual.dtype == reference.dtype
        a = actual.detach().to(device="cpu", dtype=torch.float64)
        r = reference.detach().to(device="cpu", dtype=torch.float64)
        common.update(
            {
                "shape_match": shape_match,
                "dtype_match": dtype_match,
                "actual_shape": list(actual.shape),
                "reference_shape": list(reference.shape),
                "actual_dtype": str(actual.dtype),
                "reference_dtype": str(reference.dtype),
                "actual_nonfinite_count": int((~torch.isfinite(a)).sum()),
                "reference_nonfinite_count": int((~torch.isfinite(r)).sum()),
                "total_elements": reference.numel(),
            }
        )
        if not shape_match:
            return CorrectnessResult(
                **common, status="failed", message="Output shape mismatch"
            )
        finite = torch.isfinite(a) & torch.isfinite(r)
        error = (a[finite] - r[finite]).abs()
        threshold = config.atol + config.rtol * r[finite].abs()
        failed = int((~finite).sum()) + int((error > threshold).sum())
        passed = dtype_match and failed == 0
        return CorrectnessResult(
            **common,
            status="passed" if passed else "failed",
            finite_elements=int(finite.sum()),
            failed_elements=failed,
            max_absolute_error=float(error.max()) if error.numel() else None,
            mean_absolute_error=float(error.mean()) if error.numel() else None,
            message=None
            if passed
            else (
                "Output dtype mismatch"
                if not dtype_match
                else f"{failed} of {reference.numel()} elements failed tolerance or finiteness checks"
            ),
        )
