# pyright: basic
import tiktoken
import safetensors
import torch
from torch.nn import functional as F

text = open("assets/tiny.txt")
encoding = tiktoken.encoding_for_model("gpt2")
tokens = encoding.encode(text.read())

x = torch.tensor(tokens[:64])
y_true = torch.tensor(tokens[1:65])

f = open("assets/model.safetensors", "rb")
fts = safetensors.deserialize(f.read())
param = dict()
for ft in fts:
    name = ft[0]
    data = ft[1]["data"]
    shape = ft[1]["shape"]
    param[name] = torch.frombuffer(data, dtype=torch.float32).reshape(shape)

# print(param["wte.weight"].shape)

d_model = 768
d_k = 64

wte_out = F.embedding(x, param["wte.weight"])
wpe_out = F.embedding(torch.arange(64), param["wpe.weight"])

embedding_out = wte_out + wpe_out

ln_1_in = embedding_out

for layer_i in range(12):
    ln_1_out = F.layer_norm(
        input=ln_1_in,
        normalized_shape=[d_model],
        weight=param[f"h.{layer_i}.ln_1.weight"],
        bias=param[f"h.{layer_i}.ln_1.bias"],
    )

    attn_c_attn_out = F.linear(
        input=ln_1_out,
        weight=param[f"h.{layer_i}.attn.c_attn.weight"].transpose(0, 1),
        bias=param[f"h.{layer_i}.attn.c_attn.bias"],
    )

    q, k, v = attn_c_attn_out.split(d_model, dim=1)

    attn_z_out = torch.zeros_like(ln_1_out)

    for head_i in range(12):
        a = q[:, head_i * d_k : (head_i + 1) * d_k] @ k[
            :, head_i * d_k : (head_i + 1) * d_k
        ].transpose(0, 1)
        a /= torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        mask = torch.triu(torch.ones_like(a, dtype=bool), diagonal=1)  # type: ignore
        a = torch.masked_fill(a, mask, -torch.inf)

        s = F.softmax(a, dim=1)
        attn_z_out[:, head_i * d_k : (head_i + 1) * d_k] = (
            s @ v[:, head_i * d_k : (head_i + 1) * d_k]
        )
        # exit(0)

    attn_c_proj_out = F.linear(
        input=attn_z_out,
        weight=param[f"h.{layer_i}.attn.c_proj.weight"].transpose(0, 1),
        bias=param[f"h.{layer_i}.attn.c_proj.bias"],
    )

    res_1_out = ln_1_in + attn_c_proj_out

    ln_2_out = F.layer_norm(
        input=res_1_out,
        normalized_shape=[d_model],
        weight=param[f"h.{layer_i}.ln_2.weight"],
        bias=param[f"h.{layer_i}.ln_2.bias"],
    )

    mlp_c_fc_out = F.linear(
        input=ln_2_out,
        weight=param[f"h.{layer_i}.mlp.c_fc.weight"].transpose(0, 1),
        bias=param[f"h.{layer_i}.mlp.c_fc.bias"],
    )

    mlp_gelu_out = F.gelu(mlp_c_fc_out)

    mlp_c_proj_out = F.linear(
        input=mlp_gelu_out,
        weight=param[f"h.{layer_i}.mlp.c_proj.weight"].transpose(0, 1),
        bias=param[f"h.{layer_i}.mlp.c_proj.bias"],
    )

    res_2_out = res_1_out + mlp_c_proj_out
    ln_1_in = res_2_out

ln_f_out = F.layer_norm(
    input=res_2_out,
    normalized_shape=[d_model],
    weight=param["ln_f.weight"],
    bias=param["ln_f.bias"],
)

unembedding_out = F.linear(input=ln_f_out, weight=param["wte.weight"])

token_idx = torch.argmax(unembedding_out[-1:, :], dim=-1)

print(encoding.decode(list(x)))  # type: ignore
print(encoding.decode([token_idx]))  # type: ignore

loss = F.cross_entropy(unembedding_out, y_true)
print(loss)

print(param["wte.weight"][0][0])
