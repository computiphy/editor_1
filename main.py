import click
import sys
from pathlib import Path
from src.config.loader import load_config
from src.pipeline.orchestrator import WeddingPipeline
from src.core.exceptions import ConfigValidationError

@click.command()
@click.option("--config", type=click.Path(exists=True), required=True, help="Path to YAML config file.")
@click.option("--dry-run", is_flag=True, help="Validate config and exit.")
def main(config, dry_run):
    """ Wedding Photography AI Post-Production Pipeline """
    click.echo(f"Loading config from {config}...")
    
    try:
        cfg = load_config(config)
    except Exception as e:
        click.echo(f"Error loading config: {e}", err=True)
        sys.exit(1)
        
    pipeline = WeddingPipeline(cfg)
    
    if dry_run:
        click.echo("Config validated successfully. Dry run complete.")
        return
        
    click.echo(f"Starting pipeline: {cfg.pipeline.name}")
    result = pipeline.run()
    click.echo(f"Pipeline finished in {result.elapsed_seconds:.2f} seconds.")

if __name__ == "__main__":
    main()
