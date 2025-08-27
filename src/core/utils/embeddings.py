import torch


def get_embeddings(model, tokenizer, text):
    try:
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        # Layer number (last),  batch size
        batch_hidden_states = outputs.hidden_states[-1][0].float()
        pool_embeddings = {
            "min": batch_hidden_states.min(dim=0).values.cpu().numpy().tolist(),
            "max": batch_hidden_states.max(dim=0).values.cpu().numpy().tolist(),
            "mean": batch_hidden_states.mean(dim=0).cpu().numpy().tolist(),
        }
        return pool_embeddings
    # TODO: Investigate why it fails for Qwen 3B only for specific rows
    except:
        return None
