# temp script to make star coder functional.
import torch
from transformers import AutoModelForCausalLM


hf_model_path = "Salesforce/codegen25-7b-multi"
model_name = "codegen"


# Model Wrapper
class CodeGen(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        kwargs = {"torch_dtype": torch.float32}
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            **kwargs,
        )

    def forward(self, inputs):
        op = self.model.forward(input_ids=inputs, use_cache=True)
        return_vals = []
        return_vals.append(op.logits)
        temp_past_key_values = op.past_key_values
        for item in temp_past_key_values:
            return_vals.append(item[0])
            return_vals.append(item[1])
        return tuple(return_vals)


# model pipeline
from shark.shark_importer import import_with_fx, get_f16_inputs
from transformers import AutoTokenizer
import torch_mlir
from io import BytesIO


device = "cpu"  # for GPU usage or "cpu" for CPU usage
precision = "fp16"

tokenizer = AutoTokenizer.from_pretrained(
    hf_model_path, trust_remote_code=True
)

# inputs = tokenizer.encode("def print_hello_world():", return_tensors="pt")

text = "def hello_world():"
input_ids = tokenizer(text, return_tensors="pt").input_ids

print(f"[DEBUG] input ids shape: {input_ids.shape}")

sc_model = CodeGen(hf_model_path)

op_pytorch = sc_model.forward(input_ids)
print(f"[DEBUG] {model_name} pytorch outputs generated")
for i, t in enumerate(op_pytorch):
    torch.save(t, f"{model_name}_op_pytorch_{i}.pt")

print(f"[DEBUG] pytorch outputs saved")

# import pdb; pdb.set_trace()

ts_graph = import_with_fx(
    sc_model,
    inputs=[input_ids],
    is_f16=precision == "fp16",
    f16_input_mask=[False],
    mlir_type="torchscript",
)

with open(f"{model_name}_{precision}.tsg", "w") as f_:
    f_.write(ts_graph.graph.__str__())

print(f"[DEBUG] ts graph generated")

module = torch_mlir.compile(
    ts_graph,
    [input_ids],
    torch_mlir.OutputType.LINALG_ON_TENSORS,
    use_tracing=False,
    verbose=False,
)
module = str(module)

print(f"[DEBUG] mlir module generated")
print(f"[DEBUG] converting to bytecode")
module = module.encode("UTF-8")
module = BytesIO(module)
bytecode = module.read()

print(f"[DEBUG] writing mlir to file")

with open(f"{model_name}_{precision}.mlir", "wb") as f_:
    f_.write(bytecode)


# load Mlir and compile vmfb

with open(f"{model_name}_{precision}.mlir", "rb") as f:
    bytecode = f.read()

from shark.shark_inference import SharkInference
from pathlib import Path

shark_module = SharkInference(
    mlir_module=bytecode, device=device, mlir_dialect="tm_tensor"
)

del bytecode

path = shark_module.save_module(
    Path.cwd(),
    f"{model_name}_{precision}_{device}",
    extra_args=[
        "--iree-hal-dump-executable-sources-to=ies",
        "--iree-vm-target-truncate-unsupported-floats",
        "--iree-codegen-check-ir-before-llvm-conversion=false",
        "--iree-vm-bytecode-module-output-format=flatbuffer-binary",
    ],
)
print(f"[DEBUG] Vmfb created!")
# shark_module.load_module(path)


op_shark = shark_module("forward", input_ids)
print(f"[DEBUG] shark outputs generated")
for i, t in enumerate(op_shark):
    torch.save(t, f"{model_name}_op_shark_{i}.pt")
print(f"[DEBUG] shark outpus saved")
