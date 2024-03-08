# Adapted from https://github.com/rmokady/CLIP_prefix_caption/blob/main/notebooks/clip_prefix_captioning_inference.ipynb

from pathlib import Path
from typing import Optional, Tuple

# import clip
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import nn
from tqdm import tqdm
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
)


class MLP(nn.Module):
    def __init__(self, sizes: Tuple[int, ...], bias=True, act=nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class ClipCaptionModel(nn.Module):
    # def __init__(self, prefix_length: int, prefix_size: int = 512):
    def __init__(self, prefix_length: int, prefix_size: int = 768):
        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained(
            "gpt2",
            cache_dir="/gpfsstore/rech/nry/uul79xi/huggingface",
        )
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        if prefix_length > 10:  # not enough memory
            self.clip_project = nn.Linear(
                prefix_size, self.gpt_embedding_size * prefix_length
            )
        else:
            self.clip_project = MLP(
                (
                    prefix_size,
                    (self.gpt_embedding_size * prefix_length) // 2,
                    self.gpt_embedding_size * prefix_length,
                )
            )

    # @functools.lru_cache #FIXME
    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(
            batch_size, self.prefix_length, dtype=torch.int64, device=device
        )

    def forward(
        self,
        tokens: torch.Tensor,
        prefix: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        embedding_text = self.gpt.transformer.wte(tokens)
        prefix_projections = self.clip_project(prefix).view(
            -1, self.prefix_length, self.gpt_embedding_size
        )
        # print(embedding_text.size()) #torch.Size([5, 67, 768])
        # print(prefix_projections.size()) #torch.Size([5, 1, 768])
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=mask)
        return out


class ClipCaptionPrefix(ClipCaptionModel):
    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self


def generate_beam(
    model,
    tokenizer,
    beam_size: int = 5,
    prompt=None,
    embed=None,
    entry_length=67,
    temperature=1.0,
    stop_token: str = ".",
):
    """
    Generate caption using beam search.

    Parameters
    ----------
    model: transformers.PreTrainedModel
        Model to use for the generation.

    tokenizer: transformers.PreTrainedTokenizer
        Tokenizer to use for the generation.

    beam_size: int, optional
        Beam size to use for the generation.
        Defaults to 5.

    prompt: str, optional
        Prompt to start the caption with.
        Defaults to None.

    embed: torch.Tensor, optional
        Embedding to condition the model with.
        Defaults to None.

    entry_length: int, optional
        Maximum number of words to generate.
        Defaults to 67.

    temparature: float, optional
        Temparature to use for the generation.
        Defaults to 1.0.

    stop_token: str, optional
        Token to stop the generation at.
        Defaults to ".".

    Returns
    -------
    output_texts: list of str
        Generated captions.
    """
    model.eval()
    stop_token_index = tokenizer.encode(stop_token)[0]
    tokens = None
    scores = None
    device = next(model.parameters()).device
    seq_lengths = torch.ones(beam_size, device=device)
    is_stopped = torch.zeros(beam_size, device=device, dtype=torch.bool)

    with torch.no_grad():
        # Initialize the generation with either the embeddings or the prompt
        if embed is not None:
            generated = embed
        else:
            # This is always None
            # if tokens is None:
            tokens = torch.tensor(tokenizer.encode(prompt))
            tokens = tokens.unsqueeze(0).to(device)
            generated = model.gpt.transformer.wte(tokens)

        # Run beam search
        for i in range(entry_length):
            outputs = model.gpt(inputs_embeds=generated)

            logits = outputs.logits
            logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
            logits = logits.softmax(-1).log()

            if scores is None:
                scores, next_tokens = logits.topk(beam_size, -1)
                scores = scores.squeeze(0)
                next_tokens = next_tokens.permute(1, 0)

                generated = generated.expand(beam_size, *generated.shape[1:])

                if tokens is None:
                    tokens = next_tokens
                else:
                    tokens = tokens.expand(beam_size, *tokens.shape[1:])
                    tokens = torch.cat((tokens, next_tokens), dim=1)
            else:
                logits[is_stopped] = -float(np.inf)
                logits[is_stopped, 0] = 0
                scores_sum = scores[:, None] + logits
                seq_lengths[~is_stopped] += 1
                scores_sum_average = scores_sum / seq_lengths[:, None]
                scores_sum_average, next_tokens = scores_sum_average.view(-1).topk(
                    beam_size, -1
                )
                # next_tokens_source = next_tokens // scores_sum.shape[1]
                next_tokens_source = torch.div(
                    next_tokens, scores_sum.shape[1], rounding_mode="floor"
                )
                seq_lengths = seq_lengths[next_tokens_source]
                next_tokens = next_tokens % scores_sum.shape[1]
                next_tokens = next_tokens.unsqueeze(1)
                tokens = tokens[next_tokens_source]
                tokens = torch.cat((tokens, next_tokens), dim=1)
                generated = generated[next_tokens_source]
                scores = scores_sum_average * seq_lengths
                is_stopped = is_stopped[next_tokens_source]

            next_token_embed = model.gpt.transformer.wte(next_tokens.squeeze()).view(
                generated.shape[0], 1, -1
            )

            generated = torch.cat((generated, next_token_embed), dim=1)
            is_stopped = is_stopped + next_tokens.eq(stop_token_index).squeeze()
            if is_stopped.all():
                break

    # Generate output texts
    output_list = tokens.cpu().numpy()
    output_texts = [
        tokenizer.decode(output[: int(length)])
        for output, length in zip(output_list, seq_lengths)
    ]

    # Sort output texts by score
    scores = scores / seq_lengths
    order = scores.argsort(descending=True)
    output_texts = [output_texts[i] for i in order]

    return output_texts


def generate2(
    model,
    tokenizer,
    tokens=None,
    prompt=None,
    embed=None,
    entry_count=1,
    entry_length=67,  # maximum number of words
    top_p=0.8,
    temperature=1.0,
    stop_token: str = ".",
):
    model.eval()
    # generated_num = 0
    generated_list = []
    stop_token_index = tokenizer.encode(stop_token)[0]
    filter_value = -float("Inf")
    device = next(model.parameters()).device

    with torch.no_grad():
        # for entry_idx in trange(entry_count):
        for entry_idx in range(entry_count):
            if embed is not None:
                generated = embed
            else:
                if tokens is None:
                    tokens = torch.tensor(tokenizer.encode(prompt))
                    tokens = tokens.unsqueeze(0).to(device)

                generated = model.gpt.transformer.wte(tokens)

            for i in range(entry_length):
                outputs = model.gpt(inputs_embeds=generated)
                logits = outputs.logits
                logits = logits[:, -1, :] / (temperature if temperature > 0 else 1.0)
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    nnf.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = filter_value
                next_token = torch.argmax(logits, -1).unsqueeze(0)
                next_token_embed = model.gpt.transformer.wte(next_token)
                if tokens is None:
                    tokens = next_token
                else:
                    tokens = torch.cat((tokens, next_token), dim=1)
                generated = torch.cat((generated, next_token_embed), dim=1)
                if stop_token_index == next_token.item():
                    break

            output_list = list(tokens.squeeze().cpu().numpy())
            output_text = tokenizer.decode(output_list)
            generated_list.append(output_text)

    return generated_list[0]


def generate_captions(predictions, device=torch.device("cuda:0")):
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2",
        cache_dir="/gpfsstore/rech/nry/uul79xi/huggingface",
    )

    clipcap_model_path = (
        Path("/gpfsstore/rech/nry/uul79xi/models/clipcap/coco_ViT-L14_mlp_default")
        / "coco_prefix_latest.pt"
    )
    prefix_length = 10

    clipcap = ClipCaptionModel(prefix_length)
    clipcap.load_state_dict(
        torch.load(clipcap_model_path, map_location=torch.device("cpu")), strict=False
    )
    clipcap = clipcap.to(device)

    captions = []

    clipcap.eval()
    with torch.no_grad():
        for prediction in tqdm(predictions):
            prefix = torch.from_numpy(prediction).to(device, dtype=torch.float32)
            prefix_embed = clipcap.clip_project(prefix).reshape(1, prefix_length, -1)

            # Beam search
            beam_generated_text_prefix = generate_beam(
                clipcap, tokenizer, embed=prefix_embed
            )[0]

            # Other
            other_generated_text_prefix = generate2(
                clipcap, tokenizer, embed=prefix_embed
            )

            captions.append([beam_generated_text_prefix, other_generated_text_prefix])

    return captions
