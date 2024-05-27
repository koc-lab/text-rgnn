import click
from dotenv import load_dotenv

from text_rgnn_new.sweep import add_agent, generate_sweep

load_dotenv()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.argument("dataset_name", type=click.STRING)
@click.argument("train_percentage", type=click.FLOAT)
def sweep(dataset_name: str, train_percentage: float) -> None:
    "Initialize a WandB sweep for Text-RGNN with DATASET_NAME and TRAIN_PERCENTAGE"
    sweep_id = generate_sweep(dataset_name, train_percentage)
    print(f"Created sweep with id: {sweep_id}")


@cli.command()
@click.argument("sweep_id", type=click.STRING)
def agent(sweep_id: str) -> None:
    "Attach an agent to the sweep with SWEEP_ID"
    add_agent(sweep_id)


cli()
