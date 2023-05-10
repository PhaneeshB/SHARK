import torch
import torch_mlir
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.fx.experimental.proxy_tensor import make_fx
from torch._decomp import get_decompositions
from typing import List
from io import BytesIO
from pathlib import Path
from shark.shark_downloader import download_public_file
from shark.shark_importer import transform_fx as transform_fx_
import re
import numpy as np

def get_tank_vicuna_mlir(num):
    # name can be 1 or 2 for first and second vicuna model
    mname = {1: "FirstVicuna", 2: "SecondVicuna"}
    tank_url = "gs://shark_tank/FastChat/"
    download_public_file(tank_url, mname[num])
    print(f"Downloaded model : {mname[num]} from tank")

def get_torch_mlir_module_bytecode(model, model_inputs):
    fx_g = make_fx(
        model,
        decomposition_table=get_decompositions(
            [
                torch.ops.aten.embedding_dense_backward,
                torch.ops.aten.native_layer_norm_backward,
                torch.ops.aten.slice_backward,
                torch.ops.aten.select_backward,
                torch.ops.aten.norm.ScalarOpt_dim,
                torch.ops.aten.native_group_norm,
                torch.ops.aten.upsample_bilinear2d.vec,
                torch.ops.aten.split.Tensor,
                torch.ops.aten.split_with_sizes,
            ]
        ),
    )(*model_inputs)

    print("Got FX_G")

    def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
        removed_indexes = []
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, (list, tuple)):
                    node_arg = list(node_arg)
                    node_args_len = len(node_arg)
                    for i in range(node_args_len):
                        curr_index = node_args_len - (i + 1)
                        if node_arg[curr_index] is None:
                            removed_indexes.append(curr_index)
                            node_arg.pop(curr_index)
                    node.args = (tuple(node_arg),)
                    break

        if len(removed_indexes) > 0:
            fx_g.graph.lint()
            fx_g.graph.eliminate_dead_code()
            fx_g.recompile()
        removed_indexes.sort()
        return removed_indexes


    def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
        """
        Replace tuple with tuple element in functions that return one-element tuples.
        Returns true if an unwrapping took place, and false otherwise.
        """
        print(fx_g)
        unwrapped_tuple = False
        for node in fx_g.graph.nodes:
            if node.op == "output":
                assert (
                    len(node.args) == 1
                ), "Output node must have a single argument"
                node_arg = node.args[0]
                if isinstance(node_arg, tuple):
                    if len(node_arg) == 1:
                        node.args = (node_arg[0],)
                        unwrapped_tuple = True
                        break

        if unwrapped_tuple:
            fx_g.graph.lint()
            fx_g.recompile()
        return unwrapped_tuple


    def transform_fx(fx_g):
        for node in fx_g.graph.nodes:
            if node.op == "call_function":
                if node.target in [
                    torch.ops.aten.empty,
                ]:
                    # aten.empty should be filled with zeros.
                    if node.target in [torch.ops.aten.empty]:
                        with fx_g.graph.inserting_after(node):
                            new_node = fx_g.graph.call_function(
                                torch.ops.aten.zero_,
                                args=(node,),
                            )
                            node.append(new_node)
                            node.replace_all_uses_with(new_node)
                            new_node.args = (node,)

        fx_g.graph.lint()


    transform_fx(fx_g)
    fx_g.recompile()
    removed_none_indexes = _remove_nones(fx_g)
    was_unwrapped = _unwrap_single_tuple_return(fx_g)

    fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
    fx_g.recompile()

    print("FX_G recompile")

    def strip_overloads(gm):
        """
        Modifies the target of graph nodes in :attr:`gm` to strip overloads.
        Args:
            gm(fx.GraphModule): The input Fx graph module to be modified
        """
        for node in gm.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload):
                node.target = node.target.overloadpacket
        gm.recompile()

    strip_overloads(fx_g)
    ts_g = torch.jit.script(fx_g)
    print("Got TS_G")

    return ts_g

# returns a compl
def compile_vicuna(model, model_inputs, model_name, model_vmfb_name):
    # model_name is needed to save the MLIR file
    # model_vmfb_name is needed to save/load the compiled vmfbs

    # ADD Device Arg
    from shark.shark_inference import SharkInference
    vmfb_path = Path(model_vmfb_name + ".vmfb")
    if vmfb_path.exists():
        shark_module = SharkInference(None, device="cuda", mlir_dialect="tm_tensor")
        shark_module.load_module(vmfb_path)
        return shark_module

    mlir_path = Path(model_name + ".mlir")
    print(f"[DEBUG] mlir path { mlir_path} {'exists' if mlir_path.exists() else 'does not exist'}")
    if mlir_path.exists():
        with open(mlir_path, "rb") as f:
            bytecode = f.read()
    else:
        # check online for the mlir if not, compile it on the machine with a warning
        # can use get_tank_vicuna_mlir to download mlir.

        ts_graph = get_torch_mlir_module_bytecode(model, model_inputs)
        module = torch_mlir.compile(
            ts_graph,
            [*model_inputs],
            torch_mlir.OutputType.LINALG_ON_TENSORS,
            use_tracing=False,
            verbose=False,
        )

        bytecode_stream = BytesIO()
        module.operation.write_bytecode(bytecode_stream)
        bytecode = bytecode_stream.getvalue()
    f_ = open(model_name+".mlir", "wb")
    f_.write(bytecode)
    print("Saved mlir")
    f_.close()

    shark_module = SharkInference(
        mlir_module=bytecode, device="cuda", mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    import os
    path = shark_module.save_module(
        os.getcwd(), model_vmfb_name, []
    )
    print("Saved vmfb at ", str(path))

    return shark_module

# Requires input_ids as tensor(1x40)
class FirstVicuna(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)#.cuda().half()
        print(type(self.model))
    def forward(self, input_ids, attention_mask):
        op = self.model(input_ids=input_ids, attention_mask = attention_mask, use_cache=True)
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return_vals.append(torch.sum(attention_mask).reshape([1]))
        return tuple(return_vals)

# Requires input_ids as tensor(1x1),
#          past_key_values = 32 length tuple containing tuple of tensor pairs, which is same as output
#                            of firstVicuna[1:]

class SecondVicuna(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)#.cuda().half()
    def forward(self, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47, i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63, i64, inp_size):
        #input_ids = input_tuple[0]
        #input_tuple = torch.unbind(pkv, dim=0)
        token = i0
        ret_inp_size = inp_size + 1
        inp_size = int(inp_size)
        # past_key_values = ((i1, i2), (i3, i4,), (i5, i6,), (i7, i8,), (i9, i10,), (i11, i12,), (i13, i14,), (i15, i16,), (i17, i18,), (i19, i20,), (i21, i22,), (i23, i24,), (i25, i26,), (i27, i28,), (i29, i30,), (i31, i32,), (i33, i34,), (i35, i36,), (i37, i38,), (i39, i40,), (i41, i42,), (i43, i44,), (i45, i46,), (i47, i48,), (i49, i50,), (i51, i52,), (i53, i54,), (i55, i56,), (i57, i58,), (i59, i60,), (i61, i62,), (i63, i64,))
        past_key_values = ((i1[:,:,0:inp_size,:], i2[:,:,0:inp_size,:]), (i3[:,:,0:inp_size,:], i4[:,:,0:inp_size,:],), (i5[:,:,0:inp_size,:], i6[:,:,0:inp_size,:],), (i7[:,:,0:inp_size,:], i8[:,:,0:inp_size,:],), (i9[:,:,0:inp_size,:], i10[:,:,0:inp_size,:],),
                           (i11[:,:,0:inp_size,:], i12[:,:,0:inp_size,:],), (i13[:,:,0:inp_size,:], i14[:,:,0:inp_size,:],), (i15[:,:,0:inp_size,:], i16[:,:,0:inp_size,:],), (i17[:,:,0:inp_size,:], i18[:,:,0:inp_size,:],),
                           (i19[:,:,0:inp_size,:], i20[:,:,0:inp_size,:],), (i21[:,:,0:inp_size,:], i22[:,:,0:inp_size,:],), (i23[:,:,0:inp_size,:], i24[:,:,0:inp_size,:],), (i25[:,:,0:inp_size,:], i26[:,:,0:inp_size,:],),
                           (i27[:,:,0:inp_size,:], i28[:,:,0:inp_size,:],), (i29[:,:,0:inp_size,:], i30[:,:,0:inp_size,:],), (i31[:,:,0:inp_size,:], i32[:,:,0:inp_size,:],), (i33[:,:,0:inp_size,:], i34[:,:,0:inp_size,:],),
                           (i35[:,:,0:inp_size,:], i36[:,:,0:inp_size,:],), (i37[:,:,0:inp_size,:], i38[:,:,0:inp_size,:],), (i39[:,:,0:inp_size,:], i40[:,:,0:inp_size,:],), (i41[:,:,0:inp_size,:], i42[:,:,0:inp_size,:],),
                           (i43[:,:,0:inp_size,:], i44[:,:,0:inp_size,:],), (i45[:,:,0:inp_size,:], i46[:,:,0:inp_size,:],), (i47[:,:,0:inp_size,:], i48[:,:,0:inp_size,:],), (i49[:,:,0:inp_size,:], i50[:,:,0:inp_size,:],),
                           (i51[:,:,0:inp_size,:], i52[:,:,0:inp_size,:],), (i53[:,:,0:inp_size,:], i54[:,:,0:inp_size,:],), (i55[:,:,0:inp_size,:], i56[:,:,0:inp_size,:],), (i57[:,:,0:inp_size,:], i58[:,:,0:inp_size,:],),
                           (i59[:,:,0:inp_size,:], i60[:,:,0:inp_size,:],), (i61[:,:,0:inp_size,:], i62[:,:,0:inp_size,:],), (i63[:,:,0:inp_size,:], i64[:,:,0:inp_size,:],))
        op = self.model(input_ids=token, use_cache=True, past_key_values=past_key_values)
        # return_vals = []
        # return_vals.append(op.logits)
        # temp_past_key_values = op.past_key_values
        # for item in temp_past_key_values:
        #     return_vals.append()
        #     return_vals.append()
        # return tuple(return_vals)


        i1[:,:,inp_size:inp_size+1, :] = op.past_key_values[0][0][:,:,inp_size:inp_size+1,:]
        i2[:,:,inp_size:inp_size+1, :] = op.past_key_values[0][1][:,:,inp_size:inp_size+1,:]
        i3[:,:,inp_size:inp_size+1, :] = op.past_key_values[1][0][:,:,inp_size:inp_size+1,:]
        i4[:,:,inp_size:inp_size+1, :] = op.past_key_values[1][1][:,:,inp_size:inp_size+1,:]
        i5[:,:,inp_size:inp_size+1, :] = op.past_key_values[2][0][:,:,inp_size:inp_size+1,:]
        i6[:,:,inp_size:inp_size+1, :] = op.past_key_values[2][1][:,:,inp_size:inp_size+1,:]
        i7[:,:,inp_size:inp_size+1, :] = op.past_key_values[3][0][:,:,inp_size:inp_size+1,:]
        i8[:,:,inp_size:inp_size+1, :] = op.past_key_values[3][1][:,:,inp_size:inp_size+1,:]
        i9[:,:,inp_size:inp_size+1, :] = op.past_key_values[4][0][:,:,inp_size:inp_size+1,:]
        i10[:,:,inp_size:inp_size+1, :] = op.past_key_values[4][1][:,:,inp_size:inp_size+1,:]
        i11[:,:,inp_size:inp_size+1, :] = op.past_key_values[5][0][:,:,inp_size:inp_size+1,:]
        i12[:,:,inp_size:inp_size+1, :] = op.past_key_values[5][1][:,:,inp_size:inp_size+1,:]
        i13[:,:,inp_size:inp_size+1, :] = op.past_key_values[6][0][:,:,inp_size:inp_size+1,:]
        i14[:,:,inp_size:inp_size+1, :] = op.past_key_values[6][1][:,:,inp_size:inp_size+1,:]
        i15[:,:,inp_size:inp_size+1, :] = op.past_key_values[7][0][:,:,inp_size:inp_size+1,:]
        i16[:,:,inp_size:inp_size+1, :] = op.past_key_values[7][1][:,:,inp_size:inp_size+1,:]
        i17[:,:,inp_size:inp_size+1, :] = op.past_key_values[8][0][:,:,inp_size:inp_size+1,:]
        i18[:,:,inp_size:inp_size+1, :] = op.past_key_values[8][1][:,:,inp_size:inp_size+1,:]
        i19[:,:,inp_size:inp_size+1, :] = op.past_key_values[9][0][:,:,inp_size:inp_size+1,:]
        i20[:,:,inp_size:inp_size+1, :] = op.past_key_values[9][1][:,:,inp_size:inp_size+1,:]
        i21[:,:,inp_size:inp_size+1, :] = op.past_key_values[10][0][:,:,inp_size:inp_size+1,:]
        i22[:,:,inp_size:inp_size+1, :] = op.past_key_values[10][1][:,:,inp_size:inp_size+1,:]
        i23[:,:,inp_size:inp_size+1, :] = op.past_key_values[11][0][:,:,inp_size:inp_size+1,:]
        i24[:,:,inp_size:inp_size+1, :] = op.past_key_values[11][1][:,:,inp_size:inp_size+1,:]
        i25[:,:,inp_size:inp_size+1, :] = op.past_key_values[12][0][:,:,inp_size:inp_size+1,:]
        i26[:,:,inp_size:inp_size+1, :] = op.past_key_values[12][1][:,:,inp_size:inp_size+1,:]
        i27[:,:,inp_size:inp_size+1, :] = op.past_key_values[13][0][:,:,inp_size:inp_size+1,:]
        i28[:,:,inp_size:inp_size+1, :] = op.past_key_values[13][1][:,:,inp_size:inp_size+1,:]
        i29[:,:,inp_size:inp_size+1, :] = op.past_key_values[14][0][:,:,inp_size:inp_size+1,:]
        i30[:,:,inp_size:inp_size+1, :] = op.past_key_values[14][1][:,:,inp_size:inp_size+1,:]
        i31[:,:,inp_size:inp_size+1, :] = op.past_key_values[15][0][:,:,inp_size:inp_size+1,:]
        i32[:,:,inp_size:inp_size+1, :] = op.past_key_values[15][1][:,:,inp_size:inp_size+1,:]
        i33[:,:,inp_size:inp_size+1, :] = op.past_key_values[16][0][:,:,inp_size:inp_size+1,:]
        i34[:,:,inp_size:inp_size+1, :] = op.past_key_values[16][1][:,:,inp_size:inp_size+1,:]
        i35[:,:,inp_size:inp_size+1, :] = op.past_key_values[17][0][:,:,inp_size:inp_size+1,:]
        i36[:,:,inp_size:inp_size+1, :] = op.past_key_values[17][1][:,:,inp_size:inp_size+1,:]
        i37[:,:,inp_size:inp_size+1, :] = op.past_key_values[18][0][:,:,inp_size:inp_size+1,:]
        i38[:,:,inp_size:inp_size+1, :] = op.past_key_values[18][1][:,:,inp_size:inp_size+1,:]
        i39[:,:,inp_size:inp_size+1, :] = op.past_key_values[19][0][:,:,inp_size:inp_size+1,:]
        i40[:,:,inp_size:inp_size+1, :] = op.past_key_values[19][1][:,:,inp_size:inp_size+1,:]
        i41[:,:,inp_size:inp_size+1, :] = op.past_key_values[20][0][:,:,inp_size:inp_size+1,:]
        i42[:,:,inp_size:inp_size+1, :] = op.past_key_values[20][1][:,:,inp_size:inp_size+1,:]
        i43[:,:,inp_size:inp_size+1, :] = op.past_key_values[21][0][:,:,inp_size:inp_size+1,:]
        i44[:,:,inp_size:inp_size+1, :] = op.past_key_values[21][1][:,:,inp_size:inp_size+1,:]
        i45[:,:,inp_size:inp_size+1, :] = op.past_key_values[22][0][:,:,inp_size:inp_size+1,:]
        i46[:,:,inp_size:inp_size+1, :] = op.past_key_values[22][1][:,:,inp_size:inp_size+1,:]
        i47[:,:,inp_size:inp_size+1, :] = op.past_key_values[23][0][:,:,inp_size:inp_size+1,:]
        i48[:,:,inp_size:inp_size+1, :] = op.past_key_values[23][1][:,:,inp_size:inp_size+1,:]
        i49[:,:,inp_size:inp_size+1, :] = op.past_key_values[24][0][:,:,inp_size:inp_size+1,:]
        i50[:,:,inp_size:inp_size+1, :] = op.past_key_values[24][1][:,:,inp_size:inp_size+1,:]
        i51[:,:,inp_size:inp_size+1, :] = op.past_key_values[25][0][:,:,inp_size:inp_size+1,:]
        i52[:,:,inp_size:inp_size+1, :] = op.past_key_values[25][1][:,:,inp_size:inp_size+1,:]
        i53[:,:,inp_size:inp_size+1, :] = op.past_key_values[26][0][:,:,inp_size:inp_size+1,:]
        i54[:,:,inp_size:inp_size+1, :] = op.past_key_values[26][1][:,:,inp_size:inp_size+1,:]
        i55[:,:,inp_size:inp_size+1, :] = op.past_key_values[27][0][:,:,inp_size:inp_size+1,:]
        i56[:,:,inp_size:inp_size+1, :] = op.past_key_values[27][1][:,:,inp_size:inp_size+1,:]
        i57[:,:,inp_size:inp_size+1, :] = op.past_key_values[28][0][:,:,inp_size:inp_size+1,:]
        i58[:,:,inp_size:inp_size+1, :] = op.past_key_values[28][1][:,:,inp_size:inp_size+1,:]
        i59[:,:,inp_size:inp_size+1, :] = op.past_key_values[29][0][:,:,inp_size:inp_size+1,:]
        i60[:,:,inp_size:inp_size+1, :] = op.past_key_values[29][1][:,:,inp_size:inp_size+1,:]
        i61[:,:,inp_size:inp_size+1, :] = op.past_key_values[30][0][:,:,inp_size:inp_size+1,:]
        i62[:,:,inp_size:inp_size+1, :] = op.past_key_values[30][1][:,:,inp_size:inp_size+1,:]
        i63[:,:,inp_size:inp_size+1, :] = op.past_key_values[31][0][:,:,inp_size:inp_size+1,:]
        i64[:,:,inp_size:inp_size+1, :] = op.past_key_values[31][1][:,:,inp_size:inp_size+1,:]
        return (op.logits, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16, i17, i18, i19, i20, i21, i22, i23, i24, i25, i26, i27, i28, i29, i30, i31, i32, i33, i34, i35, i36, i37, i38, i39, i40, i41, i42, i43, i44, i45, i46, i47, i48, i49, i50, i51, i52, i53, i54, i55, i56, i57, i58, i59, i60, i61, i62, i63, i64, ret_inp_size)

# need a single model
def generate_output_stream(models, tokenizer, params, device, context_len=2048,):

    ## FIX VMFB path
    #
    # # import time
    # from shark.shark_inference import SharkInference

    # max_new_tokens = 256 # also change in main

    # # make sure the path exists
    # fvic_vmfb_path = "/home/shark/disk/vic_test/fvic_dir/first_vicuna.vmfb"
    # svic_vmfb_path = "/home/shark/disk/vic_test/svic_dir/second_vicuna_27_04.vmfb"

    # # default device cuda
    # # LOAD VMFBS
    # print(f"[DEBUG] Loading pre-compiled vmfbs. device force to be Cuda")
    # fvic_model = SharkInference(mlir_module=None, device="cuda", mlir_dialect="tm_tensor")
    # print("[DEBUG] loading first vic vmfb")
    # fvic_model.load_module(fvic_vmfb_path)
    # svic_model = SharkInference(mlir_module=None, device="cuda", mlir_dialect="tm_tensor")
    # print("[DEBUG] loading second vic vmfb")
    # svic_model.load_module(svic_vmfb_path)

    fvic_model, svic_model = models
    prompt = params['prompt']

    input_ids = tokenizer(prompt).input_ids
    input_id_len = len(input_ids)

    max_new_tokens = 256 # bot reply ceil
    pad_len = max_new_tokens - input_id_len

    # get padded input and attention mask
    attention_mask = torch.ones([1,input_id_len], dtype = torch.int64)
    input_ids = torch.nn.functional.pad(torch.tensor(input_ids), (pad_len, 0), mode='constant', value=1)
    input_ids = input_ids.reshape([1,256])
    attention_mask = torch.nn.functional.pad(torch.tensor(attention_mask), (pad_len, 0), mode='constant', value=0)

    fvic_inp = (input_ids, attention_mask)
    print("[DEBUG] Running FVic ")
    out = fvic_model("forward", fvic_inp) # output from SHARK
    # out = fvic_model.forward(*fvic_inp) # output from Pytorch

    torch.save(out, "fVicOpTup.pt")
    output_ids = []
    last_token_logits = torch.tensor(out[0][0][-1])
    temperature = 0.7 # param from fastchat
    # t >- 1e-4 path
    probs = torch.softmax(last_token_logits / temperature, dim=-1)
    token = torch.tensor([[int(torch.multinomial(probs, num_samples=1))]], dtype=torch.int64)
    print(f"token : {token} decoded: {tokenizer.decode(token[0][0], skip_special_tokens=True)}")
    output_ids.append(token)
    for i in range(1,max_new_tokens):
        print("[DEBUG] Running SVic ")
        sinp = [torch.tensor([[token]],  dtype=torch.int64)]
        sinp += [op for op in out[1:]]
        sinp = tuple(sinp)
        out = svic_model("forward", sinp) # output from Shark
        # out = svic_model.forward(*sinp) # output from Pytorch
        torch.save(out, "sVicOpTup.pt")
        last_token_logits = torch.tensor(out[0][0][-1])
        temperature = 0.7 # param from fastchat
        # t >- 1e-4 path
        probs = torch.softmax(last_token_logits / temperature, dim=-1)
        token = torch.tensor([[int(torch.multinomial(probs, num_samples=1))]], dtype=torch.int64)
        print(f"[DEBUG] token : {token} decoded: {tokenizer.decode(token[0][0], skip_special_tokens=True)}")
        output_ids.append(token)
        if token == tokenizer.eos_token_id:
            stopped = True
        else:
            stopped = False
        if i == max_new_tokens - 1 or stopped:
            res_word_list = []
            for tok in output_ids:
                res_word_list.append(tokenizer.decode(tok[0][0], skip_special_tokens=True))
            # may need the res_word_list to be cropped later

    return res_word_list


if __name__ == "__main__":

    import sys
    kwargs = {"torch_dtype": torch.float32} #16
    model_path = "TheBloke/vicuna-7B-1.1-HF"
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

    prompt = input("Enter a temp Prompt to get things started: ")
    input_ids = tokenizer(prompt).input_ids
    input_id_len = len(input_ids)

    max_new_tokens = 256 # bot reply ceil
    pad_len = max_new_tokens - input_id_len

    # get padded input and attention mask
    attention_mask = torch.ones([1,input_id_len], dtype = torch.int64)
    input_ids = torch.nn.functional.pad(torch.tensor(input_ids), (pad_len, 0), mode='constant', value=1)
    input_ids = input_ids.reshape([1,256])
    attention_mask = torch.nn.functional.pad(torch.tensor(attention_mask), (pad_len, 0), mode='constant', value=0)

    firstVicuna = FirstVicuna(model_path)
    # firstVicunaInput = tuple([torch.as_tensor([input_ids])])#.cuda()
    firstVicunaInput = (input_ids, attention_mask)
    shark_first_vicuna = compile_vicuna(firstVicuna, firstVicunaInput, "first_vicuna_stdaln", "first_vicuna_stdaln")

    # get First vic output from SHARK / PYTORCH
    output_first_vicuna = shark_first_vicuna("forward", firstVicunaInput)
    # output_first_vicuna = firstVicuna.forward(*firstVicunaInput)
    # output_first_vicuna_tensor = torch.tensor(output_first_vicuna[1:])

    # Create a Token from output of first Vic
    # TODO: Recheck if this is the value needed 
    last_token_logits = output_first_vicuna[0][0][-1]
    print("SHARK firstVicuna = ", str(last_token_logits))

    temperature = 0.7
    probs = torch.softmax(torch.tensor(last_token_logits / temperature), dim=-1)
    token = torch.tensor(int(torch.multinomial(probs, num_samples=1))).reshape([1,1])
    # token = torch.ones([1,1], dtype=torch.int64)#.cuda()

    # Add for 2nd vic mlir-compile
    # firstVicunaInput = tuple([torch.as_tensor([input_ids])])#.cuda()
    secondVicunaInput = [token]
    secondVicunaInput += [torch.as_tensor(op) for op in output_first_vicuna[1:]]
    secondVicunaInputTup = tuple(secondVicunaInput)
    print(f"inputs converted to tensors: {type(len(secondVicunaInputTup))}")
    #torch.save(secondVicunaInputTup, "secondVicunaInputTup.pt")

    #sys.exit()

    secondVicuna = SecondVicuna(model_path)
    # secondVicunaInputTup = torch.load("secondVicunaInputTup.pt")
    shark_second_vicuna = compile_vicuna(secondVicuna, secondVicunaInputTup, "second_vicuna_stdaln", "second_vicuna_stdaln")

    # output_second_vicuna = secondVicuna.forward(*secondVicunaInputTup)
    output_first_vicuna_tensor = torch.tensor(output_first_vicuna[1:])

    #sys.exit()
    # secondVicuna = SecondVicuna(model_path)
    # No-context single question chatbot

    # TODO: Output a little wonking for chatbot, 
    # get it to work with correct tokens and pads 
    while True:
        # get input from user
        try:
            inp = input("What's next?\n")
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        # get output from vic

        params = {
            # "model": shark_first_vicuna, # not used in generate
            "prompt": prompt,
            "temperature": 0.7,
            "max_new_tokens": max_new_tokens,
            "stop": tokenizer.eos_token, # experimental
            # "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        # res = generate_output_stream((shark_first_vicuna, shark_second_vicuna), tokenizer, params, device="cuda")
        res = generate_output_stream((firstVicuna, secondVicuna), tokenizer, params, device="cuda")

        # write it out to gradio / wherever
        print(f"Vic: {res}")