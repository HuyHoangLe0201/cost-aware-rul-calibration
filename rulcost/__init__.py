"""rulcost: cost-aware calibration of remaining-useful-life (RUL) predictions for replacement decisions.

Given point RUL predictions at an alarm and the outcomes of completed units, choose the replacement date that minimises a linear asymmetric
cost, with a conformal (finite-sample, distribution-free) estimate of the best quantile of the ratio true RUL / predicted RUL, and handle units
that were replaced before failing (right-censored labels) with a Kaplan-Meier estimate.
"""
from .alarm import alarm_table, first_alarm
from .calibrator import CostAwareCalibrator
from .core import (asymmetric_cost, conformal_quantile, critical_level, gaussian_quantile, kaplan_meier_quantile, risk_floor, target_coverage)
from .evaluate import PolicyReport, evaluate_policies, simulate_closed_loop
from .theory import density_at_quantile, excess_cost_constant, expected_coverage, required_calibration_size

__version__ = "0.1.0"
__all__ = ["CostAwareCalibrator", "critical_level", "target_coverage", "asymmetric_cost", "conformal_quantile", "gaussian_quantile",
           "kaplan_meier_quantile", "risk_floor", "first_alarm", "alarm_table", "evaluate_policies", "PolicyReport", "simulate_closed_loop",
           "expected_coverage", "excess_cost_constant", "required_calibration_size", "density_at_quantile", "__version__"]
