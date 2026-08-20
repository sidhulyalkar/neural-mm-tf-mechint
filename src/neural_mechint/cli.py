from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .adapters import HFCausalLMAdapter
from .experiments.belief_compliance import BeliefComplianceExperiment

app = typer.Typer(no_args_is_help=True, help="Causal mechanistic interpretability laboratory.")
console = Console()


@app.command()
def inspect(
    model: str = typer.Option("google/gemma-3-270m-it", help="Hugging Face causal LM id"),
):
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


if __name__ == "__main__":
    app()
