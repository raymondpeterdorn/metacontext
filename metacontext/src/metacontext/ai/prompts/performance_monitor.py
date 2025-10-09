"""Prompt performance monitoring utilities for tracking token usage and optimization metrics.

This module provides utilities to monitor prompt performance, track token usage,
and create feedback loops for continuous optimization.
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

HIGH_TOKEN_USAGE_THRESHOLD = 1500
POOR_LIMIT_COMPLIANCE_THRESHOLD = 0.8
SLOW_PROCESSING_THRESHOLD_MS = 5000


@dataclass
class PromptMetrics:
    """Metrics for a single prompt execution."""

    template_name: str
    input_token_count: int
    input_char_count: int
    output_char_count: int
    output_word_count: int
    processing_time_ms: float
    character_limit: int | None = None
    within_limit: bool = True
    schema_validation_passed: bool = True
    timestamp: float = field(default_factory=time.time)

    @property
    def efficiency_ratio(self) -> float:
        """Calculate the efficiency ratio (output/input)."""
        if self.input_char_count == 0:
            return 0.0
        return self.output_char_count / self.input_char_count

    @property
    def limit_compliance_ratio(self) -> float:
        """Calculate compliance with character limits."""
        if not self.character_limit:
            return 1.0
        return (
            min(1.0, self.character_limit / self.output_char_count)
            if self.output_char_count > 0
            else 1.0
        )


class PerformanceMonitor:
    """Monitor and track prompt performance metrics."""

    def __init__(self, metrics_file: Path | None = None) -> None:
        """Initialize performance monitor.

        Args:
            metrics_file: Optional file to persist metrics

        """
        self.metrics: list[PromptMetrics] = []
        self.metrics_file = metrics_file
        self._load_metrics()

    def _load_metrics(self) -> None:
        """Load existing metrics from file if available."""
        if self.metrics_file and self.metrics_file.exists():
            try:
                with self.metrics_file.open() as f:
                    data = json.load(f)
                    self.metrics = [PromptMetrics(**metric) for metric in data]
            except (json.JSONDecodeError, TypeError) as e:
                logger.warning(
                    "Failed to load metrics from %s: %s",
                    self.metrics_file,
                    e,
                )

    def _save_metrics(self) -> None:
        """Save metrics to file if configured."""
        if self.metrics_file:
            try:
                self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
                with self.metrics_file.open("w") as f:
                    data = [metric.__dict__ for metric in self.metrics]
                    json.dump(data, f, indent=2)
                logger.debug(
                    "Saved %d metrics to %s",
                    len(self.metrics),
                    self.metrics_file,
                )
            except (OSError, TypeError) as e:
                logger.warning("Failed to save metrics to %s: %s", self.metrics_file, e)

    def record_prompt_execution(
        self,
        *,
        template_name: str,
        input_content: str,
        output_content: str,
        processing_time_ms: float,
        character_limit: int | None = None,
        schema_validation_passed: bool = True,
    ) -> PromptMetrics:
        """Record metrics for a prompt execution.

        Args:
            template_name: Name of the template used
            input_content: Full input prompt content
            output_content: Generated output content
            processing_time_ms: Time taken to process in milliseconds
            character_limit: Expected character limit for output
            schema_validation_passed: Whether output passed schema validation

        Returns:
            PromptMetrics object with calculated metrics

        """
        input_token_count = estimate_token_count(input_content)
        input_char_count = len(input_content)
        output_char_count = len(output_content)
        output_word_count = len(output_content.split())

        within_limit = True
        if character_limit:
            within_limit = output_char_count <= character_limit

        metrics = PromptMetrics(
            template_name=template_name,
            input_token_count=input_token_count,
            input_char_count=input_char_count,
            output_char_count=output_char_count,
            output_word_count=output_word_count,
            processing_time_ms=processing_time_ms,
            character_limit=character_limit,
            within_limit=within_limit,
            schema_validation_passed=schema_validation_passed,
        )

        self.metrics.append(metrics)
        self._save_metrics()

        return metrics

    def get_template_statistics(self, template_name: str) -> dict[str, Any]:
        """Get performance statistics for a specific template.

        Args:
            template_name: Name of the template

        Returns:
            Dictionary of performance statistics

        """
        template_metrics = [m for m in self.metrics if m.template_name == template_name]

        if not template_metrics:
            return {"error": f"No metrics found for template: {template_name}"}

        input_tokens = [m.input_token_count for m in template_metrics]
        output_chars = [m.output_char_count for m in template_metrics]
        processing_times = [m.processing_time_ms for m in template_metrics]
        compliance_rates = [m.within_limit for m in template_metrics]

        return {
            "template_name": template_name,
            "execution_count": len(template_metrics),
            "avg_input_tokens": sum(input_tokens) / len(input_tokens),
            "avg_output_chars": sum(output_chars) / len(output_chars),
            "avg_processing_time_ms": sum(processing_times) / len(processing_times),
            "limit_compliance_rate": sum(compliance_rates) / len(compliance_rates),
            "total_tokens_processed": sum(input_tokens),
            "efficiency_ratio": sum(m.efficiency_ratio for m in template_metrics)
            / len(template_metrics),
        }

    def get_overall_statistics(self) -> dict[str, Any]:
        """Get overall performance statistics across all templates.

        Returns:
            Dictionary of overall performance statistics

        """
        if not self.metrics:
            return {"error": "No metrics available"}

        template_names = list({m.template_name for m in self.metrics})

        total_input_tokens = sum(m.input_token_count for m in self.metrics)
        total_output_chars = sum(m.output_char_count for m in self.metrics)
        avg_processing_time = sum(m.processing_time_ms for m in self.metrics) / len(
            self.metrics,
        )
        overall_compliance = sum(m.within_limit for m in self.metrics) / len(
            self.metrics,
        )

        return {
            "total_executions": len(self.metrics),
            "unique_templates": len(template_names),
            "total_input_tokens": total_input_tokens,
            "total_output_chars": total_output_chars,
            "avg_processing_time_ms": avg_processing_time,
            "overall_compliance_rate": overall_compliance,
            "templates": template_names,
        }

    def identify_optimization_opportunities(self) -> list[dict[str, Any]]:
        """Identify templates that could benefit from optimization.

        Returns:
            List of optimization recommendations

        """
        recommendations = []
        template_names = list({m.template_name for m in self.metrics})

        for template_name in template_names:
            stats = self.get_template_statistics(template_name)

            # High token usage
            if stats["avg_input_tokens"] > HIGH_TOKEN_USAGE_THRESHOLD:
                recommendations.append(
                    {
                        "template": template_name,
                        "issue": "high_token_usage",
                        "current_avg": stats["avg_input_tokens"],
                        "recommendation": "Consider reducing context size or using more aggressive preprocessing",
                    },
                )

            # Poor limit compliance
            if stats["limit_compliance_rate"] < POOR_LIMIT_COMPLIANCE_THRESHOLD:
                recommendations.append(
                    {
                        "template": template_name,
                        "issue": "poor_limit_compliance",
                        "compliance_rate": stats["limit_compliance_rate"],
                        "recommendation": "Review character limits or add stricter output constraints",
                    },
                )

            # Slow processing
            if (
                stats["avg_processing_time_ms"] > SLOW_PROCESSING_THRESHOLD_MS
            ):  # 5 seconds
                recommendations.append(
                    {
                        "template": template_name,
                        "issue": "slow_processing",
                        "avg_time_ms": stats["avg_processing_time_ms"],
                        "recommendation": "Optimize prompt complexity or reduce context size",
                    },
                )

        return recommendations


def estimate_token_count(text: str) -> int:
    """Estimate token count for text content.

    Uses a simple heuristic: ~4 characters per token for English text.

    Args:
        text: Text content to estimate

    Returns:
        Estimated token count

    """
    # Simple estimation: average ~4 characters per token
    return max(1, len(text) // 4)


def create_performance_report(monitor: PerformanceMonitor) -> str:
    """Create a formatted performance report.

    Args:
        monitor: PerformanceMonitor instance

    Returns:
        Formatted report string

    """
    overall = monitor.get_overall_statistics()

    if "error" in overall:
        return "No performance data available."

    report = []
    report.append("🔍 PROMPT PERFORMANCE REPORT")
    report.append("=" * 40)
    report.append(f"Total Executions: {overall['total_executions']}")
    report.append(f"Unique Templates: {overall['unique_templates']}")
    report.append(f"Total Input Tokens: {overall['total_input_tokens']:,}")
    report.append(f"Total Output Characters: {overall['total_output_chars']:,}")
    report.append(f"Average Processing Time: {overall['avg_processing_time_ms']:.1f}ms")
    report.append(f"Overall Compliance Rate: {overall['overall_compliance_rate']:.1%}")
    report.append("")

    # Template-specific statistics
    template_names = overall["templates"]
    if template_names:
        report.append("📊 TEMPLATE PERFORMANCE")
        report.append("-" * 30)

        for template_name in template_names:
            stats = monitor.get_template_statistics(template_name)
            report.append(f"Template: {template_name}")
            report.append(f"  Executions: {stats['execution_count']}")
            report.append(f"  Avg Input Tokens: {stats['avg_input_tokens']:.0f}")
            report.append(f"  Avg Output Chars: {stats['avg_output_chars']:.0f}")
            report.append(
                f"  Avg Processing Time: {stats['avg_processing_time_ms']:.1f}ms",
            )
            report.append(f"  Compliance Rate: {stats['limit_compliance_rate']:.1%}")
            report.append("")

    # Optimization opportunities
    recommendations = monitor.identify_optimization_opportunities()
    if recommendations:
        report.append("⚠️ OPTIMIZATION OPPORTUNITIES")
        report.append("-" * 35)

        for rec in recommendations:
            report.append(f"Template: {rec['template']}")
            report.append(f"  Issue: {rec['issue']}")
            report.append(f"  Recommendation: {rec['recommendation']}")
            report.append("")

    return "\n".join(report)
