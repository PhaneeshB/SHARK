# get llama shark inference objects

import torch
from transformers import LlamaTokenizer
from apps.language_models.src.model_wrappers.llama_model import  FirstLlama
from shark.shark_importer import import_with_fx
import torch_mlir
import re
from shark.shark_inference import SharkInference
from pathlib import Path
import gc


def write_in_dynamic_inputs0(module, dynamic_input_size):
    print("[DEBUG] writing dynamic inputs to first llama")
    # Current solution for ensuring mlir files support dynamic inputs
    # TODO: find a more elegant way to implement this
    new_lines = []
    module = module.splitlines()
    while module:
        line = module.pop(0)
        line = re.sub(f"{dynamic_input_size}x", "?x", line)
        if "?x" in line:
            line = re.sub("tensor.empty\(\)", "tensor.empty(%dim)", line)
        line = re.sub(f" {dynamic_input_size},", " %dim,", line)
        if "tensor.empty" in line and "?x?" in line:
            line = re.sub(
                "tensor.empty\(%dim\)", "tensor.empty(%dim, %dim)", line
            )
        if "arith.cmpi" in line:
            line = re.sub(f"c{dynamic_input_size}", "dim", line)
        if "%0 = tensor.empty(%dim) : tensor<?xi64>" in line:
            new_lines.append(
                "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>"
            )
        if "%dim = tensor.dim %arg0, %c1 : tensor<1x?xi64>" in line:
            continue

        new_lines.append(line)
    return "\n".join(new_lines)



model_name = "llama-7b"
hf_model_path = "decapoda-research/llama-7b-hf"
tokenizer = LlamaTokenizer.from_pretrained(hf_model_path)


precision = "fp16"
weight_group_size = 128
hf_auth_token = None

save_tmlir_module = False
device = "cuda"
llama_vmfb_path = Path(f"first_{model_name}_{precision}_{device}.vmfb")

print("[DEBUG] generating mlir on device")
# Select a compilation prompt such that the resulting input_ids
# from the model's tokenizer has shape [1, 19]

compilation_prompt = "".join(["0" for _ in range(17)])

# generate first vicuna
compilation_input_ids = tokenizer(
    compilation_prompt,
    return_tensors="pt",
).input_ids
compilation_input_ids = torch.tensor(
    compilation_input_ids
).reshape([1, 19])
firstLlamaCompileInput = (compilation_input_ids,)
model = FirstLlama(
    hf_model_path,
    precision,
    weight_group_size,
    model_name,
    hf_auth_token,
)

print(f"[DEBUG] generating torchscript graph")

ts_graph = import_with_fx(
    model,
    firstLlamaCompileInput,
    is_f16=precision == "fp16",
    precision=precision,
    f16_input_mask=[False, False],
    mlir_type="torchscript",
)
del model
firstLlamaCompileInput = list(firstLlamaCompileInput)
firstLlamaCompileInput[0] = torch_mlir.TensorPlaceholder.like(
    firstLlamaCompileInput[0], dynamic_axes=[1]
)

firstLlamaCompileInput = tuple(firstLlamaCompileInput)
first_module = None
print(f"[DEBUG] generating torch mlir")

# TODO : FOR LATER
# if precision in ["int4", "int8"]:
#     first_module = torch_mlir.compile(
#         ts_graph,
#         [*firstLlamaCompileInput],
#         output_type=torch_mlir.OutputType.TORCH,
#         backend_legal_ops=[
#             "brevitas.matmul_rhs_group_quant"
#         ],
#         extra_library=brevitas_matmul_rhs_group_quant_library,
#         use_tracing=False,
#         verbose=False,
#     )
#     print(f"[DEBUG] converting torch to linalg")
#     run_pipeline_with_repro_report(
#         first_module,
#         "builtin.module(func.func(torch-unpack-torch-tensor),torch-backend-to-linalg-on-tensors-backend-pipeline)",
#         description="Lowering Torch Backend IR -> Linalg-on-Tensors Backend IR",
#     )
# else:

first_module = torch_mlir.compile(
    ts_graph,
    [*firstLlamaCompileInput],
    torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=False,
    verbose=False,
)
del ts_graph
del firstLlamaCompileInput
gc.collect()

print(
    "[DEBUG] successfully generated first Llama linalg mlir"
)
first_module = write_in_dynamic_inputs0(
    str(first_module), dynamic_input_size=19
)

if save_tmlir_module:
    with open(f"first_llama_{precision}.mlir", "w+") as f:
        f.write(first_module)

shark_module = SharkInference(
    mlir_module=first_module,
    device=device,
    mlir_dialect="tm_tensor",
)
path = shark_module.save_module(
    llama_vmfb_path.parent.absolute(),
    llama_vmfb_path.stem,
    extra_args=[
        "--iree-vm-target-truncate-unsupported-floats",
        "--iree-codegen-check-ir-before-llvm-conversion=false",
        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
    ],
)
print("Saved vic vmfb at ", str(path))
shark_module.load_module(path)


# def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
#     removed_indexes = []
#     for node in fx_g.graph.nodes:
#         if node.op == "output":
#             assert (
#                 len(node.args) == 1
#             ), "Output node must have a single argument"
#             node_arg = node.args[0]
#             if isinstance(node_arg, (list, tuple)):
#                 node_arg = list(node_arg)
#                 node_args_len = len(node_arg)
#                 for i in range(node_args_len):
#                     curr_index = node_args_len - (i + 1)
#                     if node_arg[curr_index] is None:
#                         removed_indexes.append(curr_index)
#                         node_arg.pop(curr_index)
#                 node.args = (tuple(node_arg),)
#                 break

#     if len(removed_indexes) > 0:
#         fx_g.graph.lint()
#         fx_g.graph.eliminate_dead_code()
#         fx_g.recompile()
#     removed_indexes.sort()
#     return removed_indexes

# def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
#     """
#     Replace tuple with tuple element in functions that return one-element tuples.
#     Returns true if an unwrapping took place, and false otherwise.
#     """
#     unwrapped_tuple = False
#     for node in fx_g.graph.nodes:
#         if node.op == "output":
#             assert (
#                 len(node.args) == 1
#             ), "Output node must have a single argument"
#             node_arg = node.args[0]
#             if isinstance(node_arg, tuple):
#                 if len(node_arg) == 1:
#                     node.args = (node_arg[0],)
#                     unwrapped_tuple = True
#                     break

#     if unwrapped_tuple:
#         fx_g.graph.lint()
#         fx_g.recompile()
#     return unwrapped_tuple

# def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
#     for node in fx_g.graph.nodes:
#         if node.op == "output":
#             assert (
#                 len(node.args) == 1
#             ), "Output node must have a single argument"
#             node_arg = node.args[0]
#             if isinstance(node_arg, tuple):
#                 return len(node_arg) == 0
#     return False

# def transform_fx(fx_g):
#     for node in fx_g.graph.nodes:
#         if node.op == "call_function":
#             if node.target in [
#                 torch.ops.aten.empty,
#             ]:
#                 # aten.empty should be filled with zeros.
#                 if node.target in [torch.ops.aten.empty]:
#                     with fx_g.graph.inserting_after(node):
#                         new_node = fx_g.graph.call_function(
#                             torch.ops.aten.zero_,
#                             args=(node,),
#                         )
#                         node.append(new_node)
#                         node.replace_all_uses_with(new_node)
#                         new_node.args = (node,)

#     fx_g.graph.lint()


# @make_simple_dynamo_backend
# def refbackend_torchdynamo_backend(
#     fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
# ):
#     # handling usage of empty tensor without initializing
#     transform_fx(fx_graph)
#     fx_graph.recompile()
#     if _returns_nothing(fx_graph):
#         return fx_graph
#     removed_none_indexes = _remove_nones(fx_graph)
#     was_unwrapped = _unwrap_single_tuple_return(fx_graph)

#     mlir_module = torch_mlir.compile(
#         fx_graph, example_inputs, output_type="linalg-on-tensors"
#     )

#     bytecode_stream = BytesIO()
#     mlir_module.operation.write_bytecode(bytecode_stream)
#     bytecode = bytecode_stream.getvalue()

#     shark_module = SharkInference(
#         mlir_module=bytecode, device=args.device, mlir_dialect="tm_tensor"
#     )
#     shark_module.compile()

#     def compiled_callable(*inputs):
#         inputs = [x.numpy() for x in inputs]
#         result = shark_module("forward", inputs)
#         if was_unwrapped:
#             result = [
#                 result,
#             ]
#         if not isinstance(result, list):
#             result = torch.from_numpy(result)
#         else:
#             result = tuple(torch.from_numpy(x) for x in result)
#             result = list(result)
#             for removed_index in removed_none_indexes:
#                 result.insert(removed_index, None)
#             result = tuple(result)
#         return result

#     return compiled_callable