import torch
from typing import List
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification

path_model = "/Users/lebahoai/Workspace/Techcom/ConfluenceSearch/my_model_folder_st/models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"
onnx_path = "onnx_format/model.onnx"

class DisableCompileContextManager:
    def __init__(self):
        self._original_compile = torch.compile

    def __enter__(self):
        # Turn torch.compile into a no-op
        torch.compile = lambda *args, **kwargs: lambda x: x

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.compile = self._original_compile

# Wrap model để flatten attentions
class ExportModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask):
        output = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True
        )
        # Flatten: last_hidden_state, pooler_output, rồi + tuple attentions thành list tensor riêng
        return (output.last_hidden_state, output.pooler_output) + output.attentions

with DisableCompileContextManager():
    config = AutoConfig.from_pretrained(path_model, torch_dtype=torch.float)
    model = AutoModel.from_pretrained(path_model, config=config, torch_dtype=torch.float).to("cpu")
    
    # Sử dụng wrapped model
    export_model = ExportModel(model)

    tokenizer = AutoTokenizer.from_pretrained(path_model)
    
    # Create dummy input with fixed size
    dummy_text = ["sample text", "my name is le ba hoai, i am a software engineer. I live in ho chi minh city."]
    encoded = tokenizer(
        dummy_text,
        truncation=True,
        padding=True,
        max_length=512,
        return_tensors="pt"
    )

    print(f"Input IDs shape: {encoded['input_ids'].shape}")
    output = export_model(
        input_ids=encoded['input_ids'].to("cpu"),
        attention_mask=encoded['attention_mask'].to("cpu")
    )
    # output giờ là tuple: (last_hidden_state, pooler_output, attn_layer1, attn_layer2, ..., attn_layer12)
    print(len(output))  # Nên in ra 14

    # Export to ONNX
    num_layers = config.num_hidden_layers  # 12
    output_names = ['last_hidden_state', 'pooler_output'] + [f'attention_layer_{i+1}' for i in range(num_layers)]

    torch.onnx.export(
        export_model,  # Sử dụng wrapped model
        (encoded['input_ids'], encoded['attention_mask']),
        onnx_path,
        export_params=True,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=output_names,
        opset_version=18,
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'last_hidden_state': {0: 'batch_size', 1: 'sequence_length'},
            'pooler_output': {0: 'batch_size'},
            **{f'attention_layer_{i+1}': {0: 'batch_size', 1: 'num_heads', 2: 'sequence_length', 3: 'sequence_length'} for i in range(num_layers)}
        },
    )