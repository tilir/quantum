from invoke import task


@task
def install(c):
    """Install project dependencies"""
    c.run("pip install -e .[dev]")


@task
def format(c, check=False):
    """Format code with black and isort (compatible mode)"""
    check_flag = "--check" if check else ""
    c.run(f"isort {check_flag} --profile=black experiments/ tests/ tasks.py")
    c.run(f"black {check_flag} experiments/ tests/ tasks.py")


@task
def docs(c):
    """Build and open documentation"""
    with c.cd("docs"):
        c.run("make clean")
        c.run("make html")


@task
def fix(c):
    """Automatically fix linting and formatting issues"""
    c.run("black experiments/ tests/ tasks.py")
    c.run("isort experiments/ tests/ tasks.py")
    c.run(
        "autoflake --in-place --remove-all-unused-imports --recursive experiments/ tests/"
    )
    print("Formatting and basic linting issues fixed")


@task
def test(c):
    """Run all tests with coverage"""
    c.run("mkdir -p test-reports")
    c.run(
        "pytest tests/ -v --cov=experiments "
        "--junitxml=test-reports/junit.xml "
        "--cov-report=xml:test-reports/coverage.xml"
    )


@task
def test_bell(c):
    """Run only Bell state tests"""
    c.run("mkdir -p test-reports")
    c.run(
        "pytest tests/test_bell_state.py -v " "--junitxml=test-reports/bell-junit.xml"
    )


@task
def test_qzeno(c):
    """Run only Quantum Zeno tests"""
    c.run("mkdir -p test-reports")
    c.run("pytest tests/test_qzeno.py -v " "--junitxml=test-reports/qzeno-junit.xml")


@task
def run_experiments(c, output_dir="results"):
    """Run all quantum experiments"""
    c.run(f"mkdir -p {output_dir}")
    run_bell(c, output_dir)
    run_qzeno(c, output_dir)


@task
def run_bell(c, output_dir="results"):
    """Run Bell state experiment"""
    c.run(f"mkdir -p {output_dir}")
    c.run(
        f"python -m experiments.bell_state -o {output_dir}/bell.png -c {output_dir}/circuit.svg"
    )


@task
def run_qzeno(c, output_dir="results"):
    """Run Quantum Zeno experiment"""
    c.run(f"mkdir -p {output_dir}")
    c.run(f"python -m experiments.qzeno -o {output_dir}/qzeno.png")


@task
def freeze(c):
    """Generate requirements.txt from current dependencies"""
    c.run("pip freeze --exclude-editable > requirements.txt")
