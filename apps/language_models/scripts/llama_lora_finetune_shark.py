from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer
import textwrap
from transformers import LlamaTokenizer, LlamaForCausalLM
from torch.utils.data.distributed import DistributedSampler
import os
import sys
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union
import torch_mlir
from io import BytesIO
from shark.shark_inference import SharkInference
from transformers.trainer_utils import (
    has_length,
)

from transformers.trainer_pt_utils import (
    get_model_param_count,
    IterableDatasetShard,
)
from transformers.trainer_utils import (
    TrainOutput,
    speed_metrics,
    )

# from transformers.utils import (
#     is_torch_tpu_available,
# )

from transformers.trainer_callback import (
    TrainerState,
)

if TYPE_CHECKING:
    import optuna

import time

import math
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)

import fire
import torch
from datasets import load_dataset
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import rcParams
import json

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

device = "cuda"

df = pd.read_csv("bitcoin-sentiment-tweets.csv")
# df.head()

def sentiment_score_to_name(score: float):
    if score > 0:
        return "Positive"
    elif score < 0:
        return "Negative"
    return "Neutral"

dataset_data = [
    {
        "instruction": "Detect the sentiment of the tweet.",
        "input": row_dict["Tweet"],
        "output": sentiment_score_to_name(row_dict["sent_score"])
    }
    for row_dict in df.to_dict(orient="records")
]

with open("alpaca-bitcoin-sentiment-dataset.json", "w") as f:
   json.dump(dataset_data, f)

BASE_MODEL = "decapoda-research/llama-7b-hf"

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    # load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)
tokenizer.padding_side = "left"

data = load_dataset("json", data_files="alpaca-bitcoin-sentiment-dataset.json")

CUTOFF_LEN = 256

def generate_prompt(data_point):    
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.  # noqa: E501
### Instruction:
{data_point["instruction"]}
### Input:
{data_point["input"]}
### Response:
{data_point["output"]}"""

def tokenize(prompt, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt

train_val = data["train"].train_test_split(
    test_size=200, shuffle=True, seed=42
)
train_data = (
    train_val["train"].shuffle().map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].shuffle().map(generate_and_tokenize_prompt)
)

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]

BATCH_SIZE = 128
MICRO_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 3e-4
TRAIN_STEPS = 300
OUTPUT_DIR = "experiments"

# model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)

print("[DEBUG] get peft model")
model = get_peft_model(model, config)
model.print_trainable_parameters()

training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard" 
)

data_collator = transformers.DataCollatorForSeq2Seq(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
)

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

def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert (
                len(node.args) == 1
            ), "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False

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


def refbackend_torchdynamo_backend(
    fx_graph: torch.fx.GraphModule, example_inputs: List[torch.Tensor]
):
    # handling usage of empty tensor without initializing
    transform_fx(fx_graph)
    fx_graph.recompile()
    if _returns_nothing(fx_graph):
        return fx_graph
    removed_none_indexes = _remove_nones(fx_graph)
    was_unwrapped = _unwrap_single_tuple_return(fx_graph)

    mlir_module = torch_mlir.compile(
        fx_graph, example_inputs, output_type="linalg-on-tensors"
    )

    bytecode_stream = BytesIO()
    mlir_module.operation.write_bytecode(bytecode_stream)
    bytecode = bytecode_stream.getvalue()

    shark_module = SharkInference(
        mlir_module=bytecode, device=device, mlir_dialect="tm_tensor"
    )
    shark_module.compile()

    def compiled_callable(*inputs):
        inputs = [x.numpy() if type(x) == torch.Tensor else x for x in inputs]
        result = shark_module("forward", inputs)
        if was_unwrapped:
            result = [
                result,
            ]
        if not isinstance(result, list):
            result = torch.from_numpy(result)
        else:
            result = tuple(torch.from_numpy(x) for x in result)
            result = list(result)
            for removed_index in removed_none_indexes:
                result.insert(removed_index, None)
            result = tuple(result)
        return result

    return compiled_callable


class SharkTrainer(Trainer):

    @torch._dynamo.optimize(refbackend_torchdynamo_backend, dynamic=True)
    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor | Any]) -> torch.Tensor:
        print("[DEBUG] inside TRAINING STEP")
        # dyn_call = torch._dynamo.optimize(refbackend_torchdynamo_backend)(super().training_step)
        # return dyn_call(model, inputs)
        model.train()
        inputs = self._prepare_inputs(inputs)

        loss = self.compute_loss(model, inputs)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        # self.accelerator.backward(loss)
        loss.backward()
        return loss.detach() / self.args.gradient_accumulation_steps

    # def train(self, resume_from_checkpoint: str | bool | None = None, trial: Any | Dict[str, Any] = None, ignore_keys_for_eval: List[str] | None = None, **kwargs):
    #     print("[DEBUG] inside TRAIN")
    #     return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def get_train_dataloader(self) -> DataLoader:
        print("[DEBUG] inside get_train_dataloader")
        # dyn_call = torch._dynamo.optimize(refbackend_torchdynamo_backend)(super().get_train_dataloader)
        # return dyn_call()
        return super().get_train_dataloader()
    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        print("[DEBUG] inside create_optimizer_and_scheduler")
        # dyn_call = torch._dynamo.optimize(refbackend_torchdynamo_backend)(super().create_optimizer_and_scheduler)
        # return dyn_call(num_training_steps)
        return super().create_optimizer_and_scheduler(num_training_steps)
    
    def _wrap_model(self, model, training=True, dataloader=None):
        print("[DEBUG] inside _wrap_model")
        # dyn_call = torch._dynamo.optimize(refbackend_torchdynamo_backend)(super()._wrap_model)
        # return dyn_call(model, training, dataloader)
        return super()._wrap_model(model, training, dataloader)

    def compute_loss(self, model, inputs, return_outputs=False):
        print("[DEBUG] inside compute_loss")
        dyn_call = torch._dynamo.optimize(refbackend_torchdynamo_backend, dynamic=True)(super().compute_loss)
        return dyn_call(model, inputs, return_outputs)
        # return super().compute_loss(model, inputs, return_outputs)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        # self.accelerator.free_memory()

        # should be same as batch size from training params
        self_train_batch_size = batch_size

        print("[DEBUG] " + f"Currently training with a batch size of: {self_train_batch_size}")
        # Data loader and number of training steps
        
        # TODO: Accelerate causes error with torch dynamo
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        # if args.logging_steps and args.logging_steps < 1:
        #     args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        delay_optimizer_creation = False

        # if self.is_deepspeed_enabled:
        #     self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Train!
        print("[INFO]" + "***** Running training *****")
        print("[INFO]" + f"  Num examples = {num_examples:,}")
        print("[INFO]" + f"  Num Epochs = {num_train_epochs:,}")
        print("[INFO]" + f"  Instantaneous batch size per device = {self_train_batch_size:,}")
        print("[INFO]" + f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        print("[INFO]" + f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print("[INFO]" + f"  Total optimization steps = {max_steps:,}")
        print("[INFO]" + f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        
        self.state.trial_params = None
        
        
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        # self.state.is_local_process_zero = self.is_local_process_zero()
        # self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):

            # CHECK USE of TRAIN LOADER
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            # if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
            #     self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:

                        torch.nn.utils.clip_grad_norm(
                            model.parameters(),
                            args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                print( "[WARNING] "
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        print("[INFO]" + "\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        # self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if training_arguments.should_save and self.state.best_model_checkpoint is not None and training_arguments.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    print("[INFO]" + f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs:
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        # self._memory_tracker.start()

        args = training_arguments

        self.is_in_train = True # needed in evaulation_loop and prediction loop

        if len(kwargs) > 0:
            raise TypeError(f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}.")

        self_train_batch_size = training_arguments.train_batch_size # NEED TO replace!

        return self._inner_training_loop(
            batch_size = self_train_batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )


# trainer = Trainer(
trainer = SharkTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))

# model = torch.compile(model)

print("[DEBUG] begin training")


trainer.train()


model.save_pretrained(OUTPUT_DIR)