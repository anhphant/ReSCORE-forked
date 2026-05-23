import os
import torch
import torch.nn.functional as F

from typing import List, Literal, Optional
from dataclasses import dataclass, field

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

try:
    from vllm import LLM, SamplingParams
except:
    LLM = None
    SamplingParams = None

from source.module.generate.utils import EOSReachedCriteria
from source.module.generate.base import (
    BaseGenerator,
    BaseGeneratorConfig
)


@dataclass
class LlamaGeneratorConfig(BaseGeneratorConfig):

    model_name: Optional[str] = \
        "meta-llama/Meta-Llama-3.1-8B-Instruct"

    max_total_tokens: int = 4096
    max_new_tokens: int = 64
    min_new_tokens: int = 1

    temperature: float = 0.0

    repetition_penalty: float = 1.0
    length_penalty: float = 1.0

    truncation: bool = True
    padding: bool = True

    num_return_sequences: int = 1

    stop: Optional[list] = field(
        default_factory=list
    )

    include_stop_str_in_output=True

    gpu_memory_utilization=0.8

    use_vllm=False

    eos_text=None

    gpu=None


class LlamaGenerator(BaseGenerator):

    def __init__(
        self,
        cfg=LlamaGeneratorConfig()
    ):

        super().__init__(cfg)

        if (
            self.cfg.gpu is not None
            and
            torch.cuda.is_available()
        ):

            self.device = torch.device(
                f"cuda:{self.cfg.gpu}"
            )

        else:

            self.device = torch.device(

                "cuda"

                if torch.cuda.is_available()

                else "cpu"

            )

        if self.cfg.use_vllm:

            if LLM is None:

                raise ImportError(
                    "vllm not installed"
                )

            self.model = LLM(

                model=self.cfg.model_name,

                gpu_memory_utilization=
                self.cfg.gpu_memory_utilization,

                max_model_len=
                self.cfg.max_total_tokens

            )

            self.tokenizer = \
            AutoTokenizer.from_pretrained(
                self.cfg.model_name
            )

        else:

            self.model = \
            AutoModelForCausalLM\
            .from_pretrained(

                self.cfg.model_name,

                device_map={
                    "":self.device
                }

            )

            self.model.eval()

            self.tokenizer = \
            AutoTokenizer.from_pretrained(
                self.cfg.model_name
            )

        if self.tokenizer.pad_token is None:

            self.tokenizer.pad_token = \
            self.tokenizer.eos_token

        self.tokenizer.padding_side="left"

        if self.cfg.eos_text:

            self.stopping_criteria_list=\
            EOSReachedCriteria(

                tokenizer=self.tokenizer,

                eos_text=self.cfg.eos_text

            )

        else:

            self.stopping_criteria_list=None

    @torch.no_grad()
    def _generate(
        self,
        inputs:List[str]
    ):

        if self.cfg.use_vllm:

            sampling_params=SamplingParams(

                n=
                self.cfg.num_return_sequences,

                temperature=
                self.cfg.temperature,

                repetition_penalty=
                self.cfg.repetition_penalty,

                max_tokens=
                self.cfg.max_new_tokens,

                stop=
                [self.tokenizer.eos_token]
                +
                self.cfg.stop

            )

            outputs=\
            self.model.generate(

                prompts=inputs,

                sampling_params=
                sampling_params

            )

            return [

                x.outputs[0].text

                for x

                in outputs

            ]

        model_inputs=\
        self.tokenizer(

            inputs,

            return_tensors="pt",

            max_length=
            self.cfg.max_total_tokens,

            truncation=
            self.cfg.truncation,

            padding=
            self.cfg.padding

        )

        model_inputs={

            k:v.to(self.device)

            for k,v

            in model_inputs.items()

        }

        input_len=\
        model_inputs[
            "input_ids"
        ].shape[1]

        outputs=\
        self.model.generate(

            **model_inputs,

            max_new_tokens=
            self.cfg.max_new_tokens,

            min_new_tokens=
            self.cfg.min_new_tokens,

            do_sample=
            self.cfg.temperature>0,

            temperature=
            max(
                0.01,
                self.cfg.temperature
            ),

            repetition_penalty=
            self.cfg.repetition_penalty,

            eos_token_id=
            self.tokenizer.eos_token_id,

            pad_token_id=
            self.tokenizer.pad_token_id

        )

        generated = \
        self.tokenizer.batch_decode(

            outputs[
                :,
                input_len:
            ],

            skip_special_tokens=True

        )

        del outputs
        del model_inputs

        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return generated

    @torch.no_grad()
    def _score(

        self,

        input_texts,

        output_texts,

        method="perplexity_score"

    ):

        perplexities=[]

        for inp,out in zip(

            input_texts,

            output_texts

        ):

            inp_ids=\
            self.tokenizer.encode(

                inp,

                return_tensors="pt"

            ).to(
                self.device
            )

            out_ids=\
            self.tokenizer.encode(

                out,

                return_tensors="pt",

                add_special_tokens=False

            ).to(
                self.device
            )

            ids=torch.cat(

                [

                    inp_ids,

                    out_ids

                ],

                dim=1

            )

            logits=\
            self.model(
                ids
            ).logits.squeeze(0)

            logits=\
            logits[
                inp_ids.shape[1]-1:
                -1
            ]

            labels=\
            out_ids.squeeze(0)

            ppl=\
            torch.exp(

                F.cross_entropy(

                    logits,

                    labels

                )

            )

            perplexities.append(
                ppl.item()
            )

        return perplexities