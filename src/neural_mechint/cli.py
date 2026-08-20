from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch
import typer
from rich.console import Console
from rich.table import Table

from .adapters import HFCausalLMAdapter
from .applications.grounded_rag import DEMO_SCENARIOS
from .experiments.belief_compliance import BeliefComplianceExperiment
from .mechguard import grounded_rag_profile
from .mechstate import fit_mean_difference_probe

app = typer.Typer(no_args_is_help=True, help="Observe, understand, control and improve open language models.")
console = Console()


@app.command()
def inspect(model: str = typer.Option("google/gemma-3-270m-it", help="Hugging Face causal LM id")):
    """Load a model and print the hookable architecture surface."""
    adapter = HFCausalLMAdapter.from_pretrained(model)
    console.print(f"[bold]Model[/bold] {model}")
    console.print(f"[bold]Transformer blocks[/bold] {adapter.n_layers}")
    console.print(f"[bold]Tokenizer[/bold] {adapter.tokenizer.__class__.__name__}")


@app.command("belief-compliance")
def belief_compliance(
    model: str = typer.Option("google/gemma-3-270m-it", help="Hugging Face causal LM id"),
    output: Path = typer.Option(Path("artifacts/belief_compliance_baseline.json")),
):
    """Run the zero-intervention baseline for the flagship research study."""
    adapter = HFCausalLMAdapter.from_pretrained(model)
    experiment = BeliefComplianceExperiment(adapter)
    rows = experiment.baseline_table()
    table = Table("Question", "Neutral Δlogit", "Pressured Δlogit")
    for row in rows:
        table.add_row(row["question"], f'{row["neutral_logit_diff"]:.3f}', f'{row["pressured_logit_diff"]:.3f}')
    console.print(table)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2))
    console.print(f"Saved {output}")


@app.command("deployment-profile")
def deployment_profile(name: str = typer.Argument("grounded-rag")):
    """Inspect an auditable mechanistic deployment policy."""
    if name != "grounded-rag":
        raise typer.BadParameter("currently supported profile: grounded-rag")
    profile = grounded_rag_profile()
    console.print(f"[bold]{profile.name}[/bold]: {profile.description}")
    console.print(
        f"Policy: intervene_on={profile.policy.intervene_on}, "
        f"escalate_critical_count={profile.policy.escalate_critical_count}"
    )
    for monitor in profile.monitors:
        console.print(f" • {monitor}")


@app.command("grounded-rag-scenarios")
def grounded_rag_scenarios(output: Path | None = typer.Option(None, help="Optional JSON destination")):
    """Print the reference adversarial grounding scenarios."""
    rows = [asdict(scenario) | {"prompt": scenario.prompt()} for scenario in DEMO_SCENARIOS]
    table = Table("Scenario", "Question", "False user claim")
    for row in rows:
        table.add_row(row["scenario_id"], row["question"], row["false_user_claim"])
    console.print(table)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, indent=2))
        console.print(f"Saved {output}")


@app.command("calibrate-probe")
def calibrate_probe(
    positive: Path = typer.Option(..., help="torch-saved [N,D] positive activations"),
    negative: Path = typer.Option(..., help="torch-saved [N,D] negative activations"),
    name: str = typer.Option(..., help="Interpretable signal name"),
    output: Path = typer.Option(Path("artifacts/probe.pt")),
):
    """Calibrate a transparent mean-difference deployment probe."""
    pos = torch.load(positive, map_location="cpu", weights_only=True)
    neg = torch.load(negative, map_location="cpu", weights_only=True)
    probe, report = fit_mean_difference_probe(name, pos, neg)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "name": probe.name,
            "direction": probe.direction,
            "bias": probe.bias,
            "temperature": probe.temperature,
            "source": probe.source,
            "report": asdict(report),
        },
        output,
    )
    console.print(report)
    console.print(f"Saved {output}")


if __name__ == "__main__":
    app()
