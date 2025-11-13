'''from transformers import GPT2LMHeadModel, GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2")
'''
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Use local files only
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", local_files_only=True)
model = GPT2LMHeadModel.from_pretrained("gpt2", local_files_only=True)

print("Loaded GPT-2 from local cache!")
