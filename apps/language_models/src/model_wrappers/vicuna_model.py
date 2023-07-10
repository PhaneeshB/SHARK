import torch
from transformers import AutoModelForCausalLM

from brevitas_examples.llm.llm_quant.quantize import quantize_model
from brevitas_examples.llm.llm_quant.run_utils import get_model_impl


class FirstVicuna(torch.nn.Module):
    def __init__(self, model_path, precision="fp32", weight_group_size=128):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        if precision in ["int4", "int8"]:
            print("First Vicuna applying weight quantization..")
            weight_bit_width = 4 if precision == "int4" else 8
            quantize_model(
                get_model_impl(self.model).layers,
                dtype=torch.float32,
                weight_quant_type="asym",
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float",
                weight_quant_granularity="per_group",
                weight_group_size=weight_group_size,
                quantize_weight_zero_point=False,
                input_bit_width=None,
                input_scale_type="float",
                input_param_method="stats",
                input_quant_type="asym",
                input_quant_granularity="per_tensor",
                quantize_input_zero_point=False,
                seqlen=2048,
            )
            print("Weight quantization applied.")

    def forward(self, input_ids):
        op = self.model(input_ids=input_ids, use_cache=True)
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class SecondVicuna(torch.nn.Module):
    def __init__(self, model_path, precision="fp32", weight_group_size=128):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, low_cpu_mem_usage=True, **kwargs
        )
        if precision in ["int4", "int8"]:
            print("Second Vicuna applying weight quantization..")
            weight_bit_width = 4 if precision == "int4" else 8
            quantize_model(
                get_model_impl(self.model).layers,
                dtype=torch.float32,
                weight_quant_type="asym",
                weight_bit_width=weight_bit_width,
                weight_param_method="stats",
                weight_scale_precision="float",
                weight_quant_granularity="per_group",
                weight_group_size=weight_group_size,
                quantize_weight_zero_point=False,
                input_bit_width=None,
                input_scale_type="float",
                input_param_method="stats",
                input_quant_type="asym",
                input_quant_granularity="per_tensor",
                quantize_input_zero_point=False,
                seqlen=2048,
            )
            print("Weight quantization applied.")

    def forward(
        self,
        i0,
        i1,
        i2,
        i3,
        i4,
        i5,
        i6,
        i7,
        i8,
        i9,
        i10,
        i11,
        i12,
        i13,
        i14,
        i15,
        i16,
        i17,
        i18,
        i19,
        i20,
        i21,
        i22,
        i23,
        i24,
        i25,
        i26,
        i27,
        i28,
        i29,
        i30,
        i31,
        i32,
        i33,
        i34,
        i35,
        i36,
        i37,
        i38,
        i39,
        i40,
        i41,
        i42,
        i43,
        i44,
        i45,
        i46,
        i47,
        i48,
        i49,
        i50,
        i51,
        i52,
        i53,
        i54,
        i55,
        i56,
        i57,
        i58,
        i59,
        i60,
        i61,
        i62,
        i63,
        i64,
    ):
        # input_ids = input_tuple[0]
        # input_tuple = torch.unbind(pkv, dim=0)
        token = i0
        past_key_values = (
            (i1, i2),
            (
                i3,
                i4,
            ),
            (
                i5,
                i6,
            ),
            (
                i7,
                i8,
            ),
            (
                i9,
                i10,
            ),
            (
                i11,
                i12,
            ),
            (
                i13,
                i14,
            ),
            (
                i15,
                i16,
            ),
            (
                i17,
                i18,
            ),
            (
                i19,
                i20,
            ),
            (
                i21,
                i22,
            ),
            (
                i23,
                i24,
            ),
            (
                i25,
                i26,
            ),
            (
                i27,
                i28,
            ),
            (
                i29,
                i30,
            ),
            (
                i31,
                i32,
            ),
            (
                i33,
                i34,
            ),
            (
                i35,
                i36,
            ),
            (
                i37,
                i38,
            ),
            (
                i39,
                i40,
            ),
            (
                i41,
                i42,
            ),
            (
                i43,
                i44,
            ),
            (
                i45,
                i46,
            ),
            (
                i47,
                i48,
            ),
            (
                i49,
                i50,
            ),
            (
                i51,
                i52,
            ),
            (
                i53,
                i54,
            ),
            (
                i55,
                i56,
            ),
            (
                i57,
                i58,
            ),
            (
                i59,
                i60,
            ),
            (
                i61,
                i62,
            ),
            (
                i63,
                i64,
            ),
        )
        op = self.model(
            input_ids=token, use_cache=True, past_key_values=past_key_values
        )
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


class CombinedModel(torch.nn.Module):
    def __init__(
        self,
        first_vicuna_model_path="lmsys/vicuna-7b-v1.3",
        second_vicuna_model_path="lmsys/vicuna-7b-v1.3",
    ):
        super().__init__()
        self.first_vicuna = FirstVicuna(first_vicuna_model_path)
        self.second_vicuna = SecondVicuna(second_vicuna_model_path)

    def forward(self, input_ids):
        first_output = self.first_vicuna(input_ids=input_ids, use_cache=True)
        logits = first_output[0]
        pkv = first_output[1:]

        token = torch.argmax(torch.tensor(logits)[:, -1, :], dim=1)
        token = token.to(torch.int64).reshape([1, 1])
        secondVicunaInput = (token,) + tuple(pkv)
        second_output = self.second_vicuna(secondVicunaInput)
        return second_output
