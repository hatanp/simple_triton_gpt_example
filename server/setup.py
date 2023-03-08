from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print("begin")
tokenizer = AutoTokenizer.from_pretrained("TurkuNLP/gpt3-finnish-3B")
model = AutoModelForCausalLM.from_pretrained("TurkuNLP/gpt3-finnish-3B")
out_path = r"/media/volume/triton/simple_triton_gpt_example/server/model_repository/turkunlp_gpt3/1/hf_files"
tokenizer.save_pretrained(out_path)
model.save_pretrained(out_path)
