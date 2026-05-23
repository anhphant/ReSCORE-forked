import torch
import numpy as np
import random
from tqdm import tqdm
import spacy

from source.utility.data_utils import (
    load_data_from_jsonl
)
from source.utility.system_utils import (
    seed_everything,
)
from source.module.generate.llama import (
    LlamaGenerator,
    LlamaGeneratorConfig
)
from source.module.retrieve.dense import (
    DenseRetriever,
    DenseRetrieverConfig
)
from source.module.index.index import (
    Indexer,
    IndexerConfig
)
from source.pipeline.controller import (
    PipelineController
)
from source.pipeline.state import (
    BaseState,
    QuestionState,
    AnswerState,
    DocumentState,
    ResumeState,
    ThoughtState,
    GenerateState,
)
from source.module.index.docstore import (
    Docstore,
    Document
)
from dataclasses import dataclass
import os
import json

from typing import Optional, Literal, Union, List, Dict, Any

from abc import ABC, abstractmethod

from source.pipeline.utils import (
    parse_path,
    preprocess_documents,
    filter_document,
    preprocess_retrieval_query,
    clean_wrong_json_format
)
from source.pipeline.constants import (
    DOC_DOC_DELIM,
    THOUGHT_THOUGHT_DELIM
)


class BaseOutputParser(ABC):
    
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
    
    def __call__(
        self,
        generated_text: str,
        path: List[BaseState],
    ) -> BaseState:
        
        return self.parse(
            generated_text=generated_text,
            path=path
        ) 
    
    @abstractmethod
    def parse(
        self,
        generated_text: str,
        path: List[BaseState],
    ) -> BaseState:
        
        raise NotImplementedError(
            f""
        )
        
class AnswerGenerateOutputParser(
    BaseOutputParser
):

    def parse(
        self,
        generated_text:str,
        path
    ):

        import json
        import re

        last_state_id=\
        path[-1].state_id

        generated_text=\
        generated_text.strip()

        def safe_parse():

            text=\
            generated_text

            text=\
            text.replace(
                "```json",
                ""
            )

            text=\
            text.replace(
                "```",
                ""
            )

            text=text.strip()

            try:

                data=\
                json.loads(text)

                if isinstance(
                    data,
                    dict
                ):

                    if (
                        "answer"
                        in data
                    ):

                        return str(
                            data[
                                "answer"
                            ]
                        )

                if isinstance(
                    data,
                    str
                ):

                    return data

                if isinstance(
                    data,
                    int
                ):

                    return str(data)

            except:

                pass

            m=re.search(

                r'"answer"\s*:\s*"([^"]+)',

                text,

                re.DOTALL

            )

            if m:

                return m.group(
                    1
                ).strip()

            try:

                return \
                clean_wrong_json_format(

                    generated_text,

                    'answer'

                )

            except:

                return generated_text

        answer=safe_parse()

        return AnswerState(

            parent_state_id=
            last_state_id,

            answer=
            str(answer)

        )
            
            
class ThoughtGenerateOutputParser(
    BaseOutputParser
):

    def parse(
        self,
        generated_text:str,
        path
    ):

        import json
        import re

        last_state_id=\
        path[-1].state_id

        generated_text=\
        generated_text.strip()

        def safe_parse():

            text=\
            generated_text

            text=\
            text.replace(
                "```json",
                ""
            )

            text=\
            text.replace(
                "```",
                ""
            )

            text=text.strip()

            try:

                data=\
                json.loads(text)

                if isinstance(
                    data,
                    dict
                ):

                    if (
                        "thought"
                        in data
                    ):

                        return str(
                            data[
                                "thought"
                            ]
                        )

                if isinstance(
                    data,
                    str
                ):

                    return data

                if isinstance(
                    data,
                    int
                ):

                    return str(data)

            except:

                pass

            m=re.search(

                r'"thought"\s*:\s*"([^"]+)',

                text,

                re.DOTALL

            )

            if m:

                return m.group(
                    1
                ).strip()

            try:

                return \
                clean_wrong_json_format(

                    generated_text,

                    'thought'

                )

            except:

                return generated_text

        thought=safe_parse()

        return ThoughtState(

            parent_state_id=
            last_state_id,

            thought=
            str(thought)

        )

class BasePromptGenerater(ABC):
    
    @abstractmethod
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
        self.prompt_template = None
    
    @abstractmethod
    def generate(
        self,
        path,
    ):
        raise NotImplementedError(
            f""
        )
        
    def __call__(
        self,
        path,
    ):
        return self.generate(path)


class AnswerGeneratePromptGenerator(BasePromptGenerater):
    
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
        path = cfg.answer_gen_prompt_file_path
        with open(path, 'r', encoding='utf-8') as file:
            self.prompt_template = file.read()
    
    def generate(
        self,
        path,
    ):
        question_id, question, thoughts, documents = parse_path(path)
        # Rationals to string format
        thoughts_str = THOUGHT_THOUGHT_DELIM.join(thoughts) if thoughts else 'None'
        # Document to string format
        documents = preprocess_documents(
            documents=documents,
            prompt_document_from=self.cfg.prompt_document_from,
            retrieval_count=self.cfg.retrieval_count,
            prompt_max_para_count=self.cfg.prompt_max_para_count,
        )
        documents_str = DOC_DOC_DELIM.join(
            [str(document) for document in documents]
        ) if documents else 'None'
        # Make prompt
        prompt = self.prompt_template.format(
            question=question,
            thoughts=thoughts_str,
            documents=documents_str
        )
        
        return prompt

class ThoughtGeneratePromptGenerator(BasePromptGenerater):
    
    def __init__(
        self,
        cfg
    ):
        self.cfg = cfg
        path = cfg.thought_gen_prompt_file_path
        with open(path, 'r', encoding='utf-8') as file:
            self.prompt_template = file.read()
    
    def generate(
        self,
        path,
    ):
        question_id, question, thoughts, documents = parse_path(path)
        # Rationals to string format
        thoughts_str = THOUGHT_THOUGHT_DELIM.join(thoughts) if thoughts else 'None'
        # Document to string format
        documents = preprocess_documents(
            documents=documents,
            prompt_document_from=self.cfg.prompt_document_from,
            retrieval_count=self.cfg.retrieval_count,
            prompt_max_para_count=self.cfg.prompt_max_para_count,
        )
        documents_str = DOC_DOC_DELIM.join(
            [str(document) for document in documents]
        ) if documents else 'None'
        # Make prompt
        prompt = self.prompt_template.format(
            question=question,
            thoughts=thoughts_str,
            documents=documents_str
        )
        
        return prompt

class GenerationStep:
    
    def __init__(
        self,
        cfg,
        generator,
        prompt_generator: BasePromptGenerater,
        output_parser: BaseOutputParser,
    ):
        self.cfg = cfg
        self.generator = generator
        self.prompt_generator = prompt_generator
        self.output_parser = output_parser

    def __call__(
        self,
        paths: List[List[BaseState]]
    ) -> List[BaseState]:
        
        prompts = [
            self.prompt_generator(path) 
            for path in paths
        ]
        generated_texts = self.generator(
            input_texts=prompts,
        )
        all_next_states = [
            self.output_parser(generated_text, path) 
            for generated_text, path in zip(generated_texts, paths)
        ]
        
        return all_next_states
