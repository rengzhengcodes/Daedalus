import os

import timeloopfe.v4 as tl

from . import Architecture


class Eyeriss(Architecture):
    def _evaluate(
        self, x_dict: dict, eval_problem: str, brief_print: bool = False
    ) -> float:
        if brief_print:
            print(".", end="")

        # Set up the specification
        spec = tl.Specification.from_yaml_files(
            self._spec, jinja_parse_data={"problem": eval_problem}
        )
        buf = spec.architecture.find("shared_glb")
        buf.attributes["depth"] = round(
            buf.attributes["depth"] * (2.0 ** x_dict["global_buffer_size_exp_scale"])
        )
        pe = spec.architecture.find("PE_column")
        pe.spatial.meshX = round(pe.spatial.meshX * (2.0 ** x_dict["pe_x_exp_scale"]))
        pe = spec.architecture.find("PE")
        pe.spatial.meshY = round(pe.spatial.meshY * (2.0 ** x_dict["pe_y_exp_scale"]))
        spec.mapper.search_size = 2000

        # Give each run a unique ID and run the mapper
        proc_id = f"glb_scale={2.0 ** x_dict['global_buffer_size_exp_scale']},pe_x_scale={2.0 ** x_dict['pe_x_exp_scale']},pe_y_scale={2.0 ** x_dict['pe_y_exp_scale']}"
        if brief_print:
            print(".", end="")
        else:
            print(f"Starting {proc_id}")
        out_dir = os.path.abspath(f"{os.curdir}/outputs/{proc_id}")
        tl.call_mapper(spec, output_dir=out_dir, log_to=f"{out_dir}/output.log")

        # Grab the energy from the stats file
        stats = open(f"{out_dir}/timeloop-mapper.stats.txt").read()
        stats = [l.strip() for l in stats.split("\n") if l.strip()]
        energy = float(stats[-1].split("=")[-1])
        return (
            spec.architecture.find("shared_glb").attributes["depth"],
            spec.architecture.find("PE_column").spatial.meshX,
            energy,
        )
