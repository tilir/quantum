from invoke import task

@task
def install(c):
    """Install project dependencies"""
    c.run("pip install -e .[dev]")

@task
def test(c):
    """Run tests with coverage and reports"""
    c.run("mkdir -p test-reports")
    c.run("pytest tests/ -v --cov=experiments "
          "--junitxml=test-reports/junit.xml "
          "--cov-report=xml:test-reports/coverage.xml")

@task
def run_experiments(c, output_dir="results"):
    """Run all quantum experiments"""
    c.run(f"mkdir -p {output_dir}")
    c.run(f"python -m experiments.bell_state -o {output_dir}/bell.png")
    c.run(f"python -m experiments.qzeno -o {output_dir}/qzeno.png")

@task
def freeze(c):
    """Generate requirements.txt from current dependencies"""
    c.run("pip freeze > requirements.txt")
