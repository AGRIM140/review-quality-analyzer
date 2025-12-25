import torch

def extract_attention(model, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # last layer, mean over heads
    attention = outputs.attentions[-1][0].mean(dim=0)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    scores = attention.sum(dim=0)
    scores = scores / scores.max()

    return list(zip(tokens, scores.tolist()))

