from typing import Optional
import os
import threading
import timeloopfe.v4 as tl
from tlhelper import *
import joblib

Specification = tl.Specification
THIS_SCRIPT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
ARCH_DIR = os.path.join(THIS_SCRIPT_DIR, "arch")
TOP_JINJA_PATH = os.path.join(ARCH_DIR, "top.yaml.jinja2")


def get_architecture_targets():
    targets = []
    for root, dirs, files in os.walk(ARCH_DIR):
        if "arch.yaml" in files:
            c = open(os.path.join(root, "arch.yaml")).read()
            if "version: 0.4" not in c:
                continue
            targets.append(os.path.relpath(root, ARCH_DIR))
    return sorted(targets)


def run_mapper(
    arch_target,
    problem: Optional[str] = None,
    generate_ref_outputs: Optional[bool] = False,
    remove_sparse_opts: Optional[bool] = False,
):
    # This data will be supplied when rendering the top jinja2 template
    jinja_parse_data = {"architecture": arch_target}

    if problem is None:
        problem_name = "default_problem"
    else:
        problem_name = os.path.basename(problem).split(".")[0]
        jinja_parse_data["problem"] = problem

    # Set up output directory
    if generate_ref_outputs:
        output_dir = f"{ARCH_DIR}/{arch_target}/ref_outputs/{problem_name}"
    else:
        output_dir = f"{ARCH_DIR}/{arch_target}/outputs/{problem_name}"

    print(f"\n\nRunning mapper for target {arch_target} in {output_dir}...")

    # Set up output directory
    if os.path.exists(output_dir):
        os.system(f"rm -rf {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    spec = tl.Specification.from_yaml_files(
        TOP_JINJA_PATH, jinja_parse_data=jinja_parse_data
    )

    # Used for some Sparseloop tutorials to test with/without sparse optimizations
    if remove_sparse_opts:
        remove_sparse_optimizations(spec)

    tl.call_mapper(
        spec,
        output_dir=output_dir,
        dump_intermediate_to=output_dir,
    )
    assert os.path.exists(f"{output_dir}/timeloop-mapper.stats.txt"), (
        f"Mapper did not generate expected output for {arch_target}. "
        f"Please check the logs for more details."
    )


if __name__ == "__main__":
    arch_targets = get_architecture_targets()

    # Default to the first architecture if none is specified
    arch = arch_targets[0]
    # If arch is a string, make it a list
    arch = [arch] if isinstance(arch, str) else arch

    # Put togher the list of problems to run
    problems = [None]
    problem = "alexnet"
    problem = os.path.join(THIS_SCRIPT_DIR, "layer_shapes", problem)
    if os.path.isdir(problem):
        problems = [os.path.join(problem, f) for f in os.listdir(problem)]
    else:
        problems = [problem]

    n_jobs = 16
    generate_ref_outputs = False
    remove_sparse_opts = False

    # Run parallel processes for all architectures and problems
    joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(run_mapper)(a, p, generate_ref_outputs, remove_sparse_opts)
        for a in arch
        for p in problems
    )
