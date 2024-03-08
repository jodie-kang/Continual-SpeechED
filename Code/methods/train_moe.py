#!/usr/bin/env python3

import logging
import os
import pathlib
import time
import random

import torch
from hyperpyyaml import load_hyperpyyaml
from hf_whisperMoeModelV3 import WhisperMoeDecoderLayer

import speechbrain as sb
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.utils.distributed import run_on_main
from speech_data_prepare import prepare_speech_data
from backbones import SpeechEEMoeV3, dataio_prepare_moe, profileMoe


def model_test(hparams, run_opts, locales, wer_file="wer_test.txt", acc_file="acc_test.txt"):
    """Test incrementally on the given locales.

    Arguments
    ---------
    hparams : dict
        The hyperparameters.
    run_opts : dict
        The runtime options.
    locales : list[str]
        The locales to test.
    wer_file : str
        The name of the file where WER results are saved.

    """
    # Test on base + new locales
    for locale in locales:
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_speech_data,
            kwargs={
                "locales": [locale],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )

        hparams["wer_computer"] = sb.utils.metric_stats.ErrorRateStats
        hparams["acc_computer"] = sb.utils.metric_stats.ClassificationStats

        # Set forced decoder locale
        hparams["forced_decoder_locale"] = "en"

        # Define tokenizer
        tokenizer = hparams["whisper"].tokenizer

        # Create datasets, tokenization and encoding
        _, _, test_data = dataio_prepare_moe(hparams, tokenizer)

        # Trainer initialization
        speechEE_brain = SpeechEEMoeV3(
            modules=hparams["modules"], hparams=hparams, run_opts=run_opts,
        )

        # We dynamically add the tokenizer to our brain class
        # NB: This tokenizer corresponds to the one used for Whisper
        speechEE_brain.tokenizer = tokenizer

        # Testing
        locale_folder = os.path.join(hparams["output_folder"], locale)
        os.makedirs(locale_folder, exist_ok=True)
        speechEE_brain.hparams.wer_file = os.path.join(locale_folder, wer_file)
        speechEE_brain.hparams.acc_file = os.path.join(locale_folder, acc_file)
        if hparams["skip_test"]:
            # Dummy test
            train_log_backup = speechEE_brain.hparams.train_logger.save_file
            speechEE_brain.hparams.train_logger.save_file = (
                speechEE_brain.hparams.wer_file
            ) = os.path.join(locale_folder, "tmp.txt")
            test_data.data_ids = list(test_data.data.keys())[:1]
            test_data.data = {k: test_data.data[k] for k in test_data.data_ids}
            speechEE_brain.evaluate(
                test_data,
                min_key="ACC",
                test_loader_kwargs=hparams["valid_dataloader_kwargs"],
            )
            os.remove(speechEE_brain.hparams.wer_file)
            os.remove(speechEE_brain.hparams.acc_file)
            speechEE_brain.hparams.train_logger.save_file = train_log_backup
            speechEE_brain.hparams.wer_file = os.path.join(locale_folder, wer_file)
            speechEE_brain.hparams.acc_file = os.path.join(locale_folder, acc_file)
        else:
            speechEE_brain.evaluate(
                test_data,
                min_key="ACC",
                test_loader_kwargs=hparams["valid_dataloader_kwargs"],
            )

    # MACs not 100% accurate but still useful for comparisons
    if not hparams["skip_test"]:
        try:
            profileMoe(hparams, run_opts)
        except Exception:
            logging.warning(
                "Install ptflops and torchinfo to profile the model (e.g. `pip install ptflops torchinfo`)"
            )


def train(hparams, run_opts):
    """Train incrementally on the new locales.

    Arguments
    ---------
    hparams : dict
        The hyperparameters.
    run_opts : dict
        The runtime options.

    """
    # Train on new locales
    for i, locale in enumerate(hparams["task_locales"]):
        # Multi-gpu (ddp) save data preparation
        run_on_main(
            prepare_speech_data,
            kwargs={
                "locales": [locale],
                "data_folder": hparams["data_folder"],
                "max_durations": hparams["max_durations"],
            },
        )
        # Add to Whisper tokenizer's vocabulary
        tokenizer = hparams["whisper"].tokenizer
        new_tokens = ["<event>"]
        tokenizer.add_tokens(new_tokens)

        # Freeze the whole model to avoid forgetting
        model = hparams["whisper"].model
        model.requires_grad_(False)

        # Log total number of tokens
        logging.info(
            f"Total number of tokens: {hparams['whisper'].model.decoder.embed_tokens.num_embeddings}"
        )

        # Add a new random embedding for the new language token
        hparams["whisper"].model.resize_token_embeddings(len(tokenizer))

        model_config = hparams["whisper"].model.config
        adapter_hidden_dim = hparams["adapter_hidden_dim"]
        num_local_experts = len(hparams["task_locales"])
        expert_type = hparams["expert_type"]
        num_experts_per_tok = hparams["num_experts_per_tok"]
        router_aux_loss_coef = hparams["router_aux_loss_coef"]
        hparams["whisper"].model.decoder.layers += [
            WhisperMoeDecoderLayer(hparams["whisper"].model.config, adapter_hidden_dim,
                                   num_local_experts, expert_type, num_experts_per_tok, router_aux_loss_coef) for _ in
            range(hparams["num_new_decoder_layers"])
        ]
        model = hparams["whisper"].model
        # Unfreeze layer_norm parameters
        for name, param in model.decoder.named_parameters():
            print(f'Parameter: {name}, Requires gradient: {param.requires_grad}')
        hparams["whisper"].model.decoder.embed_tokens.requires_grad_()

        # Log total number of tokens
        logging.info(
            f"Total number of tokens: {hparams['whisper'].model.decoder.embed_tokens.num_embeddings}"
        )

        # Set forced decoder locale
        hparams["forced_decoder_locale"] = "en"

        # Create datasets, tokenization and encoding
        train_data, valid_data, _ = dataio_prepare_moe(hparams, tokenizer)

        length = len(train_data)
        for old_locale in hparams["task_locales"][:i]:
            run_on_main(
                prepare_speech_data,
                kwargs={
                    "locales": [old_locale],
                    "data_folder": hparams["data_folder"],
                    "max_durations": hparams["max_durations"],
                },
            )
            old_train_data, _, _ = dataio_prepare_moe(hparams, tokenizer)
            selected_keys = random.sample(
                list(old_train_data.data.keys()),
                round(
                    min(length * hparams["replay_ratio"], len(old_train_data))
                ),
            )
            old_train_data.data = {
                k: old_train_data.data[k] for k in selected_keys
            }
            train_data.data.update(old_train_data.data)

        # Shuffle all data
        all_keys = list(train_data.data.keys())
        random.shuffle(all_keys)
        train_data.data = {k: train_data.data[k] for k in all_keys}
        train_data.data_ids = list(train_data.data.keys())

        # Trainer initialization
        checkpoint_folder = os.path.join(hparams["save_folder"], locale)
        os.makedirs(checkpoint_folder, exist_ok=True)
        hparams["checkpointer"].checkpoints_dir = pathlib.Path(
            checkpoint_folder
        )
        hparams["lr_annealing"].hyperparam_value = hparams["lr"]
        hparams["lr_annealing"].metric_values.clear()
        hparams["lr_annealing"].current_patient = 0
        speechEE_brain = SpeechEEMoeV3(
            modules=hparams["modules"],
            hparams=hparams,
            run_opts=run_opts,
            opt_class=hparams["opt_class"],
            checkpointer=hparams["checkpointer"],
        )

        # We dynamically add the tokenizer to our brain class
        # NB: This tokenizer corresponds to the one used for Whisper
        speechEE_brain.tokenizer = tokenizer

        # Training
        hparams["valid_dataloader_kwargs"].pop("ckpt_prefix", None)
        hparams["epoch_counter"].current = 0
        print(f"Length of train_data: {len(train_data)}")
        speechEE_brain.fit(
            hparams["epoch_counter"],
            train_data,
            valid_data,
            train_loader_kwargs=hparams["train_dataloader_kwargs"],
            valid_loader_kwargs=hparams["valid_dataloader_kwargs"],
        )
        # Testing
        model_test(
            hparams,
            run_opts,
            hparams["task_locales"][: i + 1],
            f"wer_test_after_{locale}.txt", f"acc_test_after_{locale}.txt",
        )


if __name__ == "__main__":
    # Command-line interface
    # hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    argv = ['../hparams/speech_ace05/moe-v3.yaml', '--device', 'cuda:2', '--seed', '0', '--num_epochs', '10',
            '--expert_type', "Adapter", "--num_experts_per_tok", "1", "--router_aux_loss_coef", "0.01",
            "--train_batch_size", "8"]
    hparams_file, run_opts, overrides = sb.parse_arguments(argv)

    # If distributed_launch=True then create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)
    hparams["train_logger"].save_file = hparams[
        "train_logger"
    ].save_file.replace(
        ".txt",
        f"_locales={','.join(hparams['task_locales'])}.txt",
    )

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )


    class CustomPaddedBatch(PaddedBatch):
        """PaddedBatch with custom padding values.

        See the documentation of `speechbrain.dataio.batch.PaddedBatch`.

        """

        def __init__(self, examples, *args, **kwargs):
            for k in ["tokens_bos", "tokens_eos"]:
                max_len = max([len(x[k]) for x in examples])
                pad_value = 0.0
                if k in ["tokens_bos"]:
                    pad_value = hparams["whisper"].tokenizer.pad_token_id
                elif k == "tokens_eos":
                    pad_value = hparams["ignore_index"]
                for example in examples:
                    x = example[k]
                    example[k] = torch.nn.functional.pad(
                        x, [0, max_len - len(x)], value=pad_value
                    )
            super().__init__(examples, *args, **kwargs)


    hparams["train_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch
    hparams["valid_dataloader_kwargs"]["collate_fn"] = CustomPaddedBatch

    # Train
    start_time = time.time()
    train(hparams, run_opts)
    duration = time.time() - start_time
    logging.info(f"Time elapsed: {duration} seconds")
