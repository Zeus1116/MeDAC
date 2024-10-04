import logging
import os
import random
import time
import json
import shutil
import re

import hydra
from omegaconf import DictConfig

from nnsmith.abstract.extension import activate_ext
from nnsmith.backends.factory import BackendFactory
from nnsmith.graph_gen import SymbolicGen, model_gen, viz
from nnsmith.logging import MGEN_LOG
from nnsmith.materialize import Model, TestCase
from nnsmith.narrow_spec import auto_opset
from nnsmith.util import hijack_patch_requires, mkdir, op_filter

def extract_tensor_datatype(input_string):
    pattern = r'<(.*?)>'  # 匹配尖括号内的任意字符（非贪婪模式）
    match = re.search(pattern, input_string)
    if match:
        return match.group(1)  # 返回匹配到的内容
    else:
        return None  # 如果没有匹配到，则返回None

def extract_prefix(input_string):
    pattern = r'^([^{\(]+)(?=[{(]|$)'
    match = re.match(pattern, input_string)
    if match:
        return match.group(1)
    else:
        return 'Unknow'

def extract_integer(input_str):
    pattern = r'v(\d+)_0'
    match = re.match(pattern, input_str)
    if match:
        integer = int(match.group(1))
        return integer
    else:
        return -1

@hydra.main(version_base=None, config_path="../config", config_name="main")
def main(cfg: DictConfig):
    # Generate a random ONNX model
    # TODO(@ganler): clean terminal outputs.
    mgen_cfg = cfg["mgen"]

    seed = random.getrandbits(32) if mgen_cfg["seed"] is None else mgen_cfg["seed"]

    MGEN_LOG.info(f"Using seed {seed}")

    # TODO(@ganler): skip operators outside of model gen with `cfg[exclude]`
    model_cfg = cfg["model"]
    ModelType = Model.init(model_cfg["type"], backend_target=cfg["backend"]["target"])
    ModelType.add_seed_setter()

    if cfg["backend"]["type"] is not None:
        factory = BackendFactory.init(
            cfg["backend"]["type"],
            target=cfg["backend"]["target"],
            optmax=cfg["backend"]["optmax"],
            parse_name=True,
        )
    else:
        factory = None

    # GENERATION
    opset = auto_opset(
        ModelType,
        factory,
        vulops=mgen_cfg["vulops"],
        grad=mgen_cfg["grad_check"],
    )
    opset = op_filter(opset, mgen_cfg["include"], mgen_cfg["exclude"])
    hijack_patch_requires(mgen_cfg["patch_requires"])
    activate_ext(opset=opset, factory=factory)

    tgen_begin = time.time()
    gen = model_gen(
        opset=opset,
        method=mgen_cfg["method"],
        seed=seed,
        max_elem_per_tensor=mgen_cfg["max_elem_per_tensor"],
        max_nodes=mgen_cfg["max_nodes"],
        timeout_ms=mgen_cfg["timeout_ms"],
        rank_choices=mgen_cfg["rank_choices"],
        dtype_choices=mgen_cfg["dtype_choices"],
    )
    tgen = time.time() - tgen_begin

    if isinstance(gen, SymbolicGen):
        MGEN_LOG.info(
            f"{len(gen.last_solution)} symbols and {len(gen.solver.assertions())} constraints."
        )

        if MGEN_LOG.getEffectiveLevel() <= logging.DEBUG:
            MGEN_LOG.debug("solution:" + ", ".join(map(str, gen.last_solution)))

    # MATERIALIZATION
    tmat_begin = time.time()
    ir = gen.make_concrete()

    MGEN_LOG.info(
        f"Generated DNN has {ir.n_var()} variables and {ir.n_compute_inst()} operators."
    )

    mkdir(mgen_cfg["save"])
    if cfg["debug"]["viz"]:
        fmt = cfg["debug"]["viz_fmt"].replace(".", "")
        viz(ir, os.path.join(mgen_cfg["save"], f"graph.{fmt}"))

    model = ModelType.from_gir(ir)
    model.refine_weights()  # either random generated or gradient-based.
    model.set_grad_check(mgen_cfg["grad_check"])
    oracle = model.make_oracle()
    tmat = time.time() - tmat_begin

    tfetch_begin = time.time()
    node_list = []
    edge_list = []
    edge_dict = {}

    for var in ir.vars:
        datatype = extract_tensor_datatype(str(ir.vars[var]))
        shape = list(ir.vars[var].shape)
        if len(shape) < 5:
            padding = 5-len(shape)
            for count in range(padding):
                shape.extend([0])
        edge_dict[str(var)] = [shape, datatype]

    for inst in ir.insts:
        node_list.append([inst.identifier, extract_prefix(str(inst.iexpr.op)), inst.iexpr.args, list(user.identifier for user in inst.users[0]), list(inst.retvals())])

    for node in node_list:
        node.append(len(node[2]))
        for from_node in node[2]:
            from_id = extract_integer(str(from_node))
            edge_list.append([from_id, node[0], edge_dict[from_node][0], edge_dict[from_node][1]])

    for node in node_list:
        del node[2:5]
        print(node)
    count = 0
    for edge in edge_list:
        edge.append(count)
        count += 1
        print(edge)

    graph_info = []
    graph_info.append(len(node_list))
    graph_info.append(len(edge_list))
    op_list = []
    for node in node_list:
        op_list.append(node[1])
    op_set = set(op_list)
    graph_info.append(list(op_set))

    data_info = []
    data_info.append({
        'node_info': node_list,
        'edge_info': edge_list,
        'graph_info': graph_info
    })
    json_filename = './nnsmith_output/data_info.json'
    with open(json_filename, 'w') as jsonfile:
        json.dump(data_info, jsonfile, indent=4)
    print('Info json file has been saved')
    tfetch = time.time() - tfetch_begin

    tsave_begin = time.time()
    testcase = TestCase(model, oracle)
    testcase.dump(root_folder=mgen_cfg["save"])
    tsave = time.time() - tsave_begin

    MGEN_LOG.info(
        f"Time:  @Generation: {tgen:.2f}s  @Materialization: {tmat:.2f}s  @Save: {tsave:.2f}s  @Fetch: {tfetch:.2f}s"
    )

    tgen, tmat, tsave, tfetch = "{:.2f}".format(tgen), "{:.2f}".format(tmat), "{:.2f}".format(tsave), "{:.2f}".format(tfetch)
    gen_time = []
    gen_time.append({
        'tgen': tgen,
        'tmat': tmat,
        'tsave': tsave,
        'tfetch': tfetch
    })
    gen_time_json_file = './nnsmith_output/gen_time.json'
    with open(gen_time_json_file, 'w') as jsonfile:
        json.dump(gen_time, jsonfile, indent=4)
    print('Time json file has been saved')
    
if __name__ == "__main__":
    main()
