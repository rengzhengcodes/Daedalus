from ..optimizers import OrthoSpace, array
import os
import timeloopfe.v4 as tl
brief_point = False

class Architecture:
    def __init__(self, dimensions: tuple, bounds: tuple, spec: str) -> None:
        assert len(dimensions) == len(bounds), "Dimensions and bounds must have the same length."
        self._dimensions: tuple = tuple(dimensions)
        self._bounds: tuple = tuple(bounds)
        self._orthospace: OrthoSpace = OrthoSpace(self._dimensions, self._bounds)
        self._spec: str = spec

    def evaluate(self, x: array, brief_print: bool=False) -> float:
        """Evaluate the architecture at point x."""
        assert self._orthospace.contains(x), "Point x is not in the search space."
        # Converts the array into a dictionary.
        x_dict = {self._dimensions[i]: x[i] for i in range(len(x))}

        if brief_print:
            print('.', end='')
        # Set up the specification
        spec = tl.Specification.from_yaml_files(self._spec)
        buf = spec.architecture.find("buffer")
        buf.attributes["depth"] = round(buf.attributes["depth"] * (2 ** x_dict["global_buffer_size_scale"]))
        pe = spec.architecture.find("PE")
        pe.spatial.meshX = round(pe.spatial.meshX * (2 ** x_dict["pe_scale"]))
        spec.mapper.search_size = 2000

        # Give each run a unique ID and run the mapper
        proc_id = f"glb_scale={2 ** x_dict['global_buffer_size_scale']},pe_scale={2 ** x_dict['pe_scale']}"
        if brief_print:
            print('.', end='')
        else:
            print(f"Starting {proc_id}")
        out_dir = f"{os.curdir}/outputs/{proc_id}"
        tl.call_mapper(spec, output_dir=out_dir, log_to=f"{out_dir}/output.log")

        # Grab the energy from the stats file
        stats = open(f"{out_dir}/timeloop-mapper.stats.txt").read()
        stats = [l.strip() for l in stats.split("\n") if l.strip()]
        energy = float(stats[-1].split("=")[-1])
        return (
            spec.architecture.find("buffer").attributes["depth"],
            spec.architecture.find("PE").spatial.meshX,
            energy,
        )