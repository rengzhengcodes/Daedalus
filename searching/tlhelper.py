import timeloopfe.v4 as tl


def remove_sparse_optimizations(spec: tl.Specification):
    """This function is used by some Sparseloop tutorials to test with/without
    sparse optimizations"""
    for s in spec.get_nodes_of_type(
        (
            tl.sparse_optimizations.ActionOptimizationList,
            tl.sparse_optimizations.RepresentationFormat,
            tl.sparse_optimizations.ComputeOptimization,
        )
    ):
        s.clear()
    return spec
