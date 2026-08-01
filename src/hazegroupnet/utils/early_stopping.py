"""Frozen quality-based early stopping used by the public training recipes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


@dataclass(frozen=True)
class EarlyStoppingConfig:
    """Thresholds for meaningful validation improvement."""

    eligibility_step: int
    earliest_stop_step: int
    patience: int = 5
    min_delta_psnr: float = 0.03
    min_delta_ssim: float = 0.0005
    psnr_guard: float = 0.03

    def validate(self, *, max_steps: int, validation_interval: int) -> None:
        if min(
            self.eligibility_step,
            self.earliest_stop_step,
            self.patience,
            max_steps,
            validation_interval,
        ) <= 0:
            raise ValueError("early-stopping steps, patience, and intervals must be positive")
        if self.eligibility_step >= max_steps:
            raise ValueError("early-stopping eligibility_step must precede max_steps")
        minimum_stop = self.eligibility_step + self.patience * validation_interval
        if self.earliest_stop_step < minimum_stop:
            raise ValueError(
                "earliest_stop_step must preserve a full post-anchor patience window"
            )
        if self.earliest_stop_step > max_steps:
            raise ValueError("earliest_stop_step cannot exceed max_steps")
        if (
            self.eligibility_step % validation_interval != 0
            or self.earliest_stop_step % validation_interval != 0
        ):
            raise ValueError("early-stopping steps must align with validation_interval")
        thresholds = (self.min_delta_psnr, self.min_delta_ssim, self.psnr_guard)
        if not all(math.isfinite(value) for value in thresholds):
            raise ValueError("early-stopping thresholds must be finite")
        if self.min_delta_psnr <= 0 or self.min_delta_ssim <= 0:
            raise ValueError("meaningful-improvement deltas must be positive")
        if self.psnr_guard < 0:
            raise ValueError("psnr_guard cannot be negative")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EarlyStoppingConfig":
        return cls(
            eligibility_step=int(value["eligibility_step"]),
            earliest_stop_step=int(value["earliest_stop_step"]),
            patience=int(value.get("patience", 5)),
            min_delta_psnr=float(value.get("min_delta_psnr", 0.03)),
            min_delta_ssim=float(value.get("min_delta_ssim", 0.0005)),
            psnr_guard=float(value.get("psnr_guard", 0.03)),
        )


@dataclass
class EarlyStoppingState:
    """Serializable single-run state for the frozen plateau rule."""

    best_observed_psnr: float = float("-inf")
    best_observed_ssim: float = float("-inf")
    significant_psnr: float | None = None
    significant_ssim: float | None = None
    bad_eligible_evaluations: int = 0
    last_step: int = -1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EarlyStoppingState":
        known = {field.name for field in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown early-stopping state fields: {sorted(unknown)}")
        return cls(**dict(value))

    def observe(
        self,
        *,
        step: int,
        psnr: float,
        ssim: float,
        config: EarlyStoppingConfig,
    ) -> dict[str, Any]:
        """Record one validation result and return the frozen decision."""

        if step <= self.last_step:
            raise ValueError(f"validation steps must increase: {step} <= {self.last_step}")
        if not math.isfinite(psnr) or not math.isfinite(ssim):
            raise ValueError("validation metrics must be finite")

        prior_best_psnr = self.best_observed_psnr
        if psnr > self.best_observed_psnr or (
            math.isclose(psnr, self.best_observed_psnr, abs_tol=1e-12, rel_tol=0.0)
            and ssim > self.best_observed_ssim
        ):
            self.best_observed_psnr = float(psnr)
            self.best_observed_ssim = float(ssim)
        self.last_step = int(step)

        meaningful = False
        established_anchor = False
        if step >= config.eligibility_step:
            if self.significant_psnr is None or self.significant_ssim is None:
                self.significant_psnr = float(psnr)
                self.significant_ssim = float(ssim)
                self.bad_eligible_evaluations = 0
                meaningful = True
                established_anchor = True
            else:
                psnr_gain = float(psnr) - self.significant_psnr
                ssim_gain = float(ssim) - self.significant_ssim
                guard_reference = max(prior_best_psnr, self.significant_psnr)
                meaningful = psnr_gain >= config.min_delta_psnr or (
                    ssim_gain >= config.min_delta_ssim
                    and float(psnr) >= guard_reference - config.psnr_guard
                )
                if meaningful:
                    self.significant_psnr = max(self.significant_psnr, float(psnr))
                    self.significant_ssim = max(self.significant_ssim, float(ssim))
                    self.bad_eligible_evaluations = 0
                else:
                    self.bad_eligible_evaluations += 1

        should_stop = (
            step >= config.earliest_stop_step
            and self.bad_eligible_evaluations >= config.patience
        )
        return {
            "meaningful_improvement": meaningful,
            "established_anchor": established_anchor,
            "bad_eligible_evaluations": self.bad_eligible_evaluations,
            "should_stop": should_stop,
        }
