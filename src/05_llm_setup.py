from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


def load_chatbot(model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return pipeline("text-generation", model=model, tokenizer=tokenizer)
