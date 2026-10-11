"""Joint-reference adaptation of the 16 codec output heads, frozen backbone.

Teacher forcing follows Qwen3-TTS 0.1.1's non-streaming core geometry.
The reference is supervised with its own transcript and Auto language prefix.
"""

import hashlib

import torch
import torch.nn.functional as F

from TTS.tts.models.qwen_experimental import ExperimentalError


def output_weights(model):
    talker = model.talker
    return [
        talker.codec_head.weight,
        *(head.weight for head in talker.code_predictor.lm_head),
    ]


def weight_digest(model):
    digest = hashlib.sha256()
    for weight in output_weights(model):
        digest.update(weight.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def frozen_features(tts, prompt, reference_text):
    model = tts.model
    model.eval().requires_grad_(False)
    talker, config = model.talker, model.config
    predictor = talker.code_predictor
    codes = prompt[0].ref_code.detach().clone().long()
    speaker = prompt[0].ref_spk_embedding.detach().clone()
    if codes.ndim != 2 or codes.shape[1] != 16 or not codes.shape[0]:
        raise ExperimentalError("training_failed")
    ids = tts._tokenize_texts([tts._build_assistant_text(reference_text)])[0]
    with torch.no_grad():
        bos, eos, pad = talker.text_projection(
            talker.get_text_embeddings()(
                torch.tensor(
                    [
                        [
                            config.tts_bos_token_id,
                            config.tts_eos_token_id,
                            config.tts_pad_token_id,
                        ]
                    ]
                )
            )
        ).chunk(3, dim=1)
        # Auto describes the reference; the request language belongs to output only.
        codec_tag = talker.get_input_embeddings()(
            torch.tensor(
                [
                    [
                        config.talker_config.codec_nothink_id,
                        config.talker_config.codec_think_bos_id,
                        config.talker_config.codec_think_eos_id,
                    ]
                ]
            )
        )
        codec_prefix = torch.cat(
            [
                codec_tag,
                speaker.reshape(1, 1, -1),
                talker.get_input_embeddings()(
                    torch.tensor([[config.talker_config.codec_pad_id]])
                ),
            ],
            dim=1,
        )
        role = talker.text_projection(talker.get_text_embeddings()(ids[:, :3]))
        tags = (
            torch.cat([pad.expand(-1, codec_prefix.shape[1] - 1, -1), bos], dim=1)
            + codec_prefix
        )
        text = torch.cat(
            [talker.text_projection(talker.get_text_embeddings()(ids[:, 3:-5])), eos],
            dim=1,
        )
        text = text + talker.get_input_embeddings()(
            torch.full(text.shape[:2], config.talker_config.codec_pad_id)
        )
        codec_bos = pad + talker.get_input_embeddings()(
            torch.tensor([[config.talker_config.codec_bos_id]])
        )
        prefix = torch.cat([role, tags, text, codec_bos], dim=1)
        acoustic = talker.get_input_embeddings()(codes[:, 0])
        for index in range(1, 16):
            acoustic = acoustic + predictor.get_input_embeddings()[index - 1](
                codes[:, index]
            )
        inputs = torch.cat([prefix, acoustic.unsqueeze(0) + pad], dim=1)
        hidden = talker.model(
            inputs_embeds=inputs,
            attention_mask=torch.ones(inputs.shape[:2], dtype=torch.long),
            use_cache=False,
        ).last_hidden_state
        # State preceding each frame predicts codec0; final state predicts EOS.
        first_hidden = hidden[0, prefix.shape[1] - 1 :].detach().clone()
        first_targets = torch.cat(
            [codes[:, 0], torch.tensor([config.talker_config.codec_eos_token_id])]
        )
        sub_inputs = [
            first_hidden[:-1].unsqueeze(1),
            talker.get_input_embeddings()(codes[:, :1]),
        ]
        for index in range(1, 15):
            sub_inputs.append(
                predictor.get_input_embeddings()[index - 1](codes[:, index : index + 1])
            )
        sub_inputs = predictor.small_to_mtp_projection(torch.cat(sub_inputs, dim=1))
        residual_hidden = (
            predictor.model(inputs_embeds=sub_inputs, use_cache=False)
            .last_hidden_state.detach()
            .clone()
        )
    return first_hidden, first_targets, residual_hidden, codes


def adapt_collection(tts, references):
    model = tts.model
    model.eval().requires_grad_(False)
    talker = model.talker
    predictor = talker.code_predictor
    features = [frozen_features(tts, prompt, text) for prompt, text in references]
    if not features:
        raise ExperimentalError("training_failed")
    weights = output_weights(model)
    for weight in weights:
        weight.requires_grad_(True)
    before = weight_digest(model)
    optimizer = torch.optim.AdamW(weights, lr=2e-5, weight_decay=0)
    losses = []
    for _ in range(8):
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for first_hidden, first_targets, residual_hidden, codes in features:
            first_loss = F.cross_entropy(talker.codec_head(first_hidden), first_targets)
            residual_loss = torch.stack(
                [
                    F.cross_entropy(
                        predictor.lm_head[index - 1](residual_hidden[:, index]),
                        codes[:, index],
                    )
                    for index in range(1, 16)
                ]
            ).mean()
            loss = (first_loss + 0.3 * residual_loss) / len(features)
            if not torch.isfinite(loss):
                raise ExperimentalError("training_failed")
            loss.backward()
            step_loss += float(loss.detach())
            del loss, first_loss, residual_loss
        norm = torch.nn.utils.clip_grad_norm_(weights, 1.0)
        if not torch.isfinite(norm):
            raise ExperimentalError("training_failed")
        optimizer.step()
        losses.append(step_loss)
    after = weight_digest(model)
    if after == before or any(not torch.isfinite(weight).all() for weight in weights):
        raise ExperimentalError("training_failed")
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval().requires_grad_(False)
    return {
        "steps": len(losses),
        "pairs": len(features),
        "losses": losses,
        "trainable_parameters": sum(weight.numel() for weight in weights),
        "before": before,
        "after": after,
    }
