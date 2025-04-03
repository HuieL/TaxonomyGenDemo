import contextlib
import torch
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch_scatter import scatter
from torch_geometric.data import Batch
from src.model.gnn import load_gnn_model
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100

class ConceptGraphLLM(torch.nn.Module):

    def __init__(
        self,
        args,
        **kwargs
    ):
        super().__init__()
        self.max_txt_len = args.max_txt_len
        self.max_new_tokens = args.max_new_tokens

        print('Loading LLAMA')
        kwargs = {
            "max_memory": {0: '20GiB', 1: '20GiB', 2: '20GiB', 3: '20GiB'},
            "device_map": "auto",
            "revision": "main",
        }

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=False, revision=kwargs["revision"])
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'

        model = AutoModelForCausalLM.from_pretrained(
            args.llm_model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            **kwargs
        )

        if args.llm_frozen == 'True':
            print("Freezing LLAMA!")
            for _, param in model.named_parameters():
                param.requires_grad = False
        else:
            print("Training LLAMA with LORA!")
            model = prepare_model_for_kbit_training(model)
            lora_r: int = 8
            lora_alpha: int = 16
            lora_dropout: float = 0.05
            lora_target_modules = [
                "q_proj",
                "v_proj",
            ]
            config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, config)

        self.model = model
        print('Finish loading LLAMA!')

        self.graph_encoder = load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            hidden_channels=args.gnn_hidden_dim,
            out_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        ).to(self.model.device)

        # If you are using llama2-13b, replace with nn.Linear(2048, 5120) ...
        self.projector = nn.Sequential(
            nn.Linear(args.gnn_hidden_dim, 2048),
            nn.Sigmoid(),
            nn.Linear(2048, 4096),
        ).to(self.model.device)

        self.word_embedding = self.model.model.get_input_embeddings()

    @property
    def device(self):
        return list(self.parameters())[0].device

    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_concept_graph(self, graph):
        if graph.x.device != self.device:
            graph = graph.to(self.device)
            
        node_embeds, _ = self.graph_encoder(
            graph.x, 
            graph.edge_index.long(),
            None  
        )
        
        graph_batch = graph.batch if hasattr(graph, 'batch') else torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)
        graph_embeds = scatter(node_embeds, graph_batch, dim=0, reduce='mean')
        graph_embeds = self.projector(graph_embeds)
        
        return graph_embeds

    def prepare_paper_texts(self, concept_graph):
        paper_texts = []
        for i in range(len(concept_graph.title)):
            title = concept_graph.title[i]
            abstract = concept_graph.abstract[i] if concept_graph.abstract[i] != "N/A" else ""
            
            text = f"Paper: {title}\n"
            if abstract:
                # Truncate long abstracts
                abstract_snippet = abstract[:200] + "..." if len(abstract) > 200 else abstract
                text += f"Abstract: {abstract_snippet}\n"
                
            paper_texts.append(text)
            
        return "\n".join(paper_texts)

    def forward(self, samples):
        # Extract concept graph and prompt information
        concept_graph = samples['graph']
        instructions = samples['instruction']
        target_labels = samples['target_label']
        
        paper_text = self.prepare_paper_texts(concept_graph)
        graph_embeds = self.encode_concept_graph(concept_graph)
        
        paper_tokens = self.tokenizer(paper_text, add_special_tokens=False)
        instruction_tokens = self.tokenizer(instructions, add_special_tokens=False)
        target_tokens = self.tokenizer(target_labels, add_special_tokens=False)
        
        # Encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_size = len(instructions)
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        for i in range(batch_size):
            # Add bos & eos token
            label_input_ids = target_tokens.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            
            # Combine paper info and instruction
            paper_ids = paper_tokens.input_ids[i][:self.max_txt_len//2]  # Allow room for instruction
            instruction_ids = instruction_tokens.input_ids[i][:self.max_txt_len//2]
            input_ids = paper_ids + instruction_ids + eos_user_tokens.input_ids + label_input_ids
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # Insert graph embedding after BOS token
            if i < len(graph_embeds):
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            else:
                inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])
            
            # For labels, ignore all tokens except the target sequence
            prefix_len = inputs_embeds.shape[0] - len(label_input_ids)
            label_input_ids = [IGNORE_INDEX] * prefix_len + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            if pad_length > 0:
                batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
                batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
                batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids,
            )

        return outputs.loss

    def inference(self, samples):
        # Extract concept graph and prompt information
        concept_graph = samples['graph']
        instructions = samples['instruction']
        
        # Get paper info as text
        paper_text = self.prepare_paper_texts(concept_graph)
        graph_embeds = self.encode_concept_graph(concept_graph)
        paper_tokens = self.tokenizer(paper_text, add_special_tokens=False)
        instruction_tokens = self.tokenizer(instructions, add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)

        batch_size = len(instructions)
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        for i in range(batch_size):
            paper_ids = paper_tokens.input_ids[i][:self.max_txt_len//2] 
            instruction_ids = instruction_tokens.input_ids[i][:self.max_txt_len//2]
            input_ids = paper_ids + instruction_ids + eos_user_tokens.input_ids
            
            inputs_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            # Insert graph embedding after BOS token
            if i < len(graph_embeds):
                inputs_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), inputs_embeds], dim=0)
            else:
                inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=0)

            batch_inputs_embeds.append(inputs_embeds)
            batch_attention_mask.append([1] * inputs_embeds.shape[0])

        # pad inputs_embeds
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            if pad_length > 0:
                batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
                batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

        inputs_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                use_cache=True  # IMPORTANT!
            )
            
        # Decode generated text
        generated_concepts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract only the generated part after instruction
        final_concepts = []
        for concept in generated_concepts:
            # Find the position after instruction end
            inst_end_pos = concept.find(EOS_USER)
            if inst_end_pos != -1:
                final_concepts.append(concept[inst_end_pos + len(EOS_USER):].strip())
            else:
                final_concepts.append(concept)

        return {
            'id': samples.get('id', list(range(len(instructions)))),
            'pred': final_concepts,
            'instruction': instructions,
        }

    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        return trainable_params, all_param
