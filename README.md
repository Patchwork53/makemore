
# makemore

makemore takes one text file as input, where each line is assumed to be one training thing, and generates more things like it. Under the hood, it is an autoregressive character-level language model, with a wide choice of models from bigrams all the way to a Transformer (exactly as seen in GPT). For example, we can feed it a database of names, and makemore will generate cool baby name ideas that all sound name-like, but are not already existing names. Or if we feed it a database of company names then we can generate new ideas for a name of a company. Or we can just feed it valid scrabble words and generate english-like babble.

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

# KV Cache
An inference-time technique that makes attention O(n) by storing past keys and values. Trade memory for time.

### Speed Improvement: 
147.1 seconds without KV cache --> 20.1 seconds with KV cache <br>
<br>
Generated 4000 (shakespeare) lines of upto upto 77 characters


# Observation
- KV cache won't work during training since weights are changing
- Works with all types of embeddings
- Won't work when the context window is shifted since the KV cache in memory would be invalid since they use the old postional embeddings
- Basically works within one context window

# Upnext
- RoPE
- Speculative Decoding


