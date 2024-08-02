# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import subprocess
from datetime import datetime
from typing import List, Optional, Tuple

N_PROC_PER_NODE = 8

DEFAULT_FLAGS = {
    # MEMORY
    "optimizer_in_bwd": {
        "values": [False],
        "allowed_recipes": [
            "full_finetune_single_device",
            "lora_finetune_single_device",
        ],
    },
    "enable_activation_checkpointing": {
        "values": [False],
    },
    "fsdp_cpu_offload": {
        "values": [False],
        "allowed_recipes": ["lora_finetune_distributed", "full_finetune_distributed"],
    },
    "optimizer._component_": {
        "values": ["torch.optim.AdamW"],
    },
    # SPEED
    "fsdp_sharding_strategy": {
        "values": ["FULL_SHARD"],
        "allowed_recipes": ["lora_finetune_distributed", "full_finetune_distributed"],
    },
    "compile": {
        "values": [False],
    },
    "memory_efficient_fsdp_wrap": {
        "values": [True],
        "allowed_recipes": ["full_finetune_distributed"],
    },
    # LoRA
    "model.lora_attn_modules": {
        "values": [["q_proj", "v_proj", "k_proj", "output_proj"]],
        "allowed_recipes": ["lora_finetune_distributed", "lora_finetune_single_device"],
    },
    "model.apply_lora_to_mlp": {
        "values": [True],
        "allowed_recipes": ["lora_finetune_distributed", "lora_finetune_single_device"],
    },
    "model.apply_lora_to_output": {
        "values": [True],
        "allowed_recipes": ["lora_finetune_distributed", "lora_finetune_single_device"],
    },
    "model.lora_rank": {
        "values": [16],
        "allowed_recipes": ["lora_finetune_distributed", "lora_finetune_single_device"],
    },
    "model.lora_alpha": {
        "values": [32],
        "allowed_recipes": ["lora_finetune_distributed", "lora_finetune_single_device"],
    },
    # DATASET
    "dataset.source": {
        "values": ["Yukang/LongAlpaca-12k"],
    },
    "dataset.packed": {
        "values": [True],
    },
    "dataset.split": {
        "values": ["train[:10%]"],
    },
    # LOGGING
    "metric_logger": {
        "values": ["torchtune.utils.metric_logging.WandBLogger"],
    },
    "metric_logger.project": {
        "values": ["recipe_profiling"],
    },
    "log_every_n_steps": {
        "values": [1],
    },
    "log_peak_memory_stats": {
        "values": [True],
    },
    "save_last_checkpoint": {"values": [False]},
    # Trainer
    "gradient_accumulation_steps": {
        "values": [1],
    },
    "max_steps_per_epoch": {
        "values": [10],
    },
    "epochs": {
        "values": [1],
    },
}


SPEED_FLAGS = {
    "compile": {
        "values": [True],
    },
    "fsdp_sharding_strategy": {
        "values": ["NO_SHARD"],
        "allowed_recipes": ["lora_finetune_distributed", "full_finetune_distributed"],
    },
    "memory_efficient_fsdp_wrap": {
        "values": [False],
        "allowed_recipes": ["full_finetune_distributed"],
    },
}

MEMORY_FLAGS = {
    "optimizer_in_bwd": {
        "values": [True],
        "allowed_recipes": [
            "full_finetune_single_device",
            "lora_finetune_single_device",
        ],
    },
    "enable_activation_checkpointing": {
        "values": [True],
    },
    "fsdp_cpu_offload": {
        "values": [True],
        "allowed_recipes": ["lora_finetune_distributed", "full_finetune_distributed"],
    },
    "optimizer._component_": {
        "values": ["bitsandbytes.optim.PagedAdamW8bit"],
    },
    "minimal_lora": {
        "is_multiple_flags": True,
        "values": [
            "rank_8__qv_proj_only"
        ],  # TODO: this is used just to make it easy to write the run_name
        "components": {
            "model.lora_attn_modules": {
                "values": [["q_proj", "v_proj"]],
                "allowed_recipes": [
                    "lora_finetune_distributed",
                    "lora_finetune_single_device",
                ],
            },
            "model.apply_lora_to_mlp": {
                "values": [False],
                "allowed_recipes": [
                    "lora_finetune_distributed",
                    "lora_finetune_single_device",
                ],
            },
            "model.apply_lora_to_output": {
                "values": [False],
                "allowed_recipes": [
                    "lora_finetune_distributed",
                    "lora_finetune_single_device",
                ],
            },
            "model.lora_rank": {
                "values": [8],
                "allowed_recipes": [
                    "lora_finetune_distributed",
                    "lora_finetune_single_device",
                ],
            },
            "model.lora_alpha": {
                "values": [16],
                "allowed_recipes": [
                    "lora_finetune_distributed",
                    "lora_finetune_single_device",
                ],
            },
        },
    },
}

YAML_NAMES_DICT = {
    "llama3_1/8B_full": "full_finetune_distributed",
    "llama3_1/8B_full_single_device": "full_finetune_single_device",
    "llama3_1/8B_lora": "lora_finetune_distributed",
    "llama3_1/8B_qlora_single_device": "lora_finetune_single_device",
    "llama3_1/8B_lora_single_device": "lora_finetune_single_device",
}

RECIPES_CLI = {
    "lora_finetune_distributed": "tune run --nproc_per_node {n_proc_per_node} lora_finetune_distributed --config {config}",
    "full_finetune_distributed": "tune run --nproc_per_node {n_proc_per_node} full_finetune_distributed --config {config}",
    "lora_finetune_single_device": "tune run lora_finetune_single_device --config {config}",
    "full_finetune_single_device": "tune run full_finetune_single_device --config {config}",
}


# TODO add asserts
# assert every config has the right format
# assert everything in speed/memory is in default
# assert every recipe cli is in yaml names dict


def run_command(command):
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        raise Exception(
            f"Command failed with exit code {result.returncode}: {result.stderr}"
        )


def update_config(config_dict: dict, new_flags: dict, recipe_type: str) -> dict:
    """
    Updates config_dict with new_flags {key:value} that are allowed
    for the recipe_type.
    """
    # e.g. {"optimizer_in_bwd": {"values": [False], "allowed_recipes": ["full_finetune_single_device"]}}
    for flag_key, flag_dict in new_flags.items():

        # check if this flag applies to the recipe
        no_recipe_restriction = "allowed_recipes" not in flag_dict
        if no_recipe_restriction or recipe_type in flag_dict["allowed_recipes"]:

            if flag_dict.get("is_multiple_flags", False):
                # call the function recurseely to add a dictionary of flags
                # for example, lora needs to change many flags at once (is_multiple_flags)
                config_dict = update_config(
                    config_dict,
                    new_flags=flag_dict["components"],
                    recipe_type=recipe_type,
                )
            else:
                # add to the config_dict the flag and its value, e.g. {optimizer_in_bwd: False}
                config_dict[flag_key] = flag_dict["values"][0]

    return config_dict


def run_experiments(
    yaml_name: str,
    recipe_type: str,
    default_flags: dict,
    iterate_on: dict,
    experiment_tags: List[str],
    run_name_tags: Optional[dict] = None,
    optimization_flags: Optional[dict] = None,
) -> dict:

    failed_combinations = []

    batch_size_flag = "batch_size"
    max_seq_len_flag = "dataset.max_seq_len"
    assert (batch_size_flag in iterate_on) and (
        max_seq_len_flag in iterate_on
    ), f"Couldn't find {batch_size_flag} or {max_seq_len_flag} in {iterate_on=}"

    # e.g. iterate_on={"batch_size": [1,2]}
    for batch_size_value in iterate_on[batch_size_flag]:
        oom_failed_max_seq_len = False
        for max_seq_len_value in iterate_on[max_seq_len_flag]:

            # initialize config dict with default values
            config_dict = {}
            config_dict = update_config(
                config_dict, new_flags=default_flags, recipe_type=recipe_type
            )

            # apply given optimization flags
            if optimization_flags:
                config_dict = update_config(
                    config_dict, new_flags=optimization_flags, recipe_type=recipe_type
                )

            # update with iterate_value
            config_dict[batch_size_flag] = batch_size_value
            config_dict[max_seq_len_flag] = max_seq_len_value

            # update run_name
            run_name = f"{yaml_name}__{batch_size_value}__{max_seq_len_value}"
            processed_run_name_tags = []
            if run_name_tags:
                for key, value in run_name_tags.items():
                    run_name_tag = f"{key}-{value}"
                    run_name = f"{run_name}__{run_name_tag}"
                    processed_run_name_tags.append(run_name_tag)
            run_name = f"{run_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

            config_dict["metric_logger.name"] = run_name

            # add tag
            config_dict["metric_logger.tags"] = experiment_tags
            config_dict["run_tag"] = processed_run_name_tags

            # create CLI
            cli_command = RECIPES_CLI[recipe_type].format(
                config=yaml_name, n_proc_per_node=N_PROC_PER_NODE
            )
            for key, value in config_dict.items():
                if isinstance(value, list):
                    # Join the list items into a string formatted as a list without spaces after commas
                    value = f"\"[{','.join(value)}]\""
                cli_command += f" \\\n{key}={value}"

            # run command
            try:
                print(cli_command)
                run_command(cli_command)
            except Exception as e:
                failed_combination = {
                    "yaml_name": yaml_name,
                    "recipe_type": recipe_type,
                    "batch_size": batch_size_value,
                    "max_seq_len": max_seq_len_value,
                    "config_dict": config_dict,
                    "error": e,
                }
                failed_combinations.append(failed_combination)
                oom_failed_max_seq_len = True
                print(
                    "!!!ATTENTION!!! FAILED COMMAND WITH COMBINATION: {failed_combination}"
                )

            if oom_failed_max_seq_len:
                break

    print(f"{failed_combinations=}")  # TODO: make it a logger
    return failed_combinations


def profile_config(
    yaml_name: str,
    recipe_type: str,
    experiment_tag: str,
    seq_len_combinations: Tuple[int],
) -> dict:

    all_failed_combinations = {}

    # ------DEFAULT CONFIG------
    # iterate over bsz + seq_length with default values
    default_iterate_on = {
        "dataset.max_seq_len": seq_len_combinations,
        "batch_size": [1, 2],
    }
    default_failed_combinations = run_experiments(
        yaml_name=yaml_name,
        recipe_type=recipe_type,
        default_flags=DEFAULT_FLAGS,
        iterate_on=default_iterate_on,
        experiment_tags=[experiment_tag, "default_config"],
        run_name_tags={"config": "Default"},
        optimization_flags=None,
    )
    all_failed_combinations["default_failed_combinations"] = default_failed_combinations

    # ------MEMORY OPTIMIZED CONFIG------
    # iterate over bsz + seq_length with memory optimized
    memory_iterate_on = {
        "dataset.max_seq_len": seq_len_combinations,
        "batch_size": [1, 2],
    }
    memory_failed_combinations = run_experiments(
        yaml_name=yaml_name,
        recipe_type=recipe_type,
        default_flags=DEFAULT_FLAGS,
        iterate_on=memory_iterate_on,
        experiment_tags=[experiment_tag, "memory_optimized_config"],
        run_name_tags={"config": "AllMemoryOptimization"},
        optimization_flags=MEMORY_FLAGS,
    )
    all_failed_combinations["memory_failed_combinations"] = memory_failed_combinations

    # ------SPEED OPTIMIZED CONFIG------
    # iterate over bsz + seq_length with speed optimized flags
    speed_iterate_on = {"dataset.max_seq_len": seq_len_combinations, "batch_size": [1]}
    speed_failed_combinations = run_experiments(
        yaml_name=yaml_name,
        recipe_type=recipe_type,
        default_flags=DEFAULT_FLAGS,
        iterate_on=speed_iterate_on,
        experiment_tags=[experiment_tag, "speed_optimized_config"],
        run_name_tags={"config": "AllSpeedOptimization"},
        optimization_flags=SPEED_FLAGS,
    )
    all_failed_combinations["speed_failed_combinations"] = speed_failed_combinations

    # ------ONE_BY_ONE OPTIMIZATION------
    # go over failed combinations and find min_failed_seq_len for bsz==1
    min_failed_seq_len = 1_000_000  # dummy value
    found_failed = False
    for combination in default_failed_combinations:
        if combination["batch_size"] == 1:
            min_failed_seq_len = min(combination["max_seq_len"], min_failed_seq_len)
            found_failed = True

    # get the max_seq_len right before the failed one. E.g. if it failed on 16k, get 8k
    # we will use this to test each memory/speed flag at a time.
    if not found_failed:
        max_successful_seq_len = default_iterate_on["dataset.max_seq_len"][-1]
    else:
        index_min_failed_seq_len = default_iterate_on["dataset.max_seq_len"].index(
            min_failed_seq_len
        )
        max_successful_seq_len = default_iterate_on["dataset.max_seq_len"][
            index_min_failed_seq_len - 1
        ]

    min_successful_seq_len = default_iterate_on["dataset.max_seq_len"][0]

    # run memory/speed flags one by one, so we can see their delta,
    # for only bsz=1 and 2 max_seq_len, one small and one large
    one_by_one_iterate_on = {
        "dataset.max_seq_len": list(
            set([min_successful_seq_len, max_successful_seq_len])
        ),  # remove duplicates
        "batch_size": [1],
    }

    one_by_one_failed_combinations = []
    for flags in [MEMORY_FLAGS, SPEED_FLAGS]:
        for flag_key, flag_value in flags.items():
            one_by_one_dict = {flag_key: flag_value}
            failed_combinations = run_experiments(
                yaml_name=yaml_name,
                recipe_type=recipe_type,
                default_flags=DEFAULT_FLAGS,
                iterate_on=one_by_one_iterate_on,
                experiment_tags=[experiment_tag, "one_by_one_config", flag_key],
                run_name_tags={flag_key: flag_value["values"][0]},
                optimization_flags=one_by_one_dict,
            )
            one_by_one_failed_combinations.append(failed_combinations)

    all_failed_combinations[
        "one_by_one_failed_combinations"
    ] = one_by_one_failed_combinations

    return all_failed_combinations


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Process some yaml names and recipe types."
    )
    parser.add_argument(
        "--yaml_name",
        "-n",
        type=str,
        help='The name of the yaml file. E.g. "llama3_1/8B_full_single_device"',
    )
    parser.add_argument(
        "--recipe_type",
        "-r",
        type=str,
        help='The type of the recipe. E.g. "full_finetune_single_device"',
    )
    parser.add_argument(
        "--tag",
        "-t",
        type=str,
        help='Optional tag for the experiment, added to "{exp_tag}_{yaml_name}_{recipe_type}_{now}"',
    )
    args = parser.parse_args()

    # use all configs if specific config is not given
    if args.yaml_name and args.recipe_type:
        yaml_names_dict = {args.yaml_name: args.recipe_type}
    else:
        yaml_names_dict = YAML_NAMES_DICT

    # run experiments
    for yaml_name, recipe_type in yaml_names_dict.items():
        try:
            now = datetime.now().strftime("%Y_%m_%d_%H_%M")
            experiment_tag = f"{yaml_name}_{now}"
            if args.tag:
                experiment_tag = f"{args.tag}_{experiment_tag}"

            all_failed_combinations = profile_config(
                yaml_name,
                recipe_type,
                experiment_tag=experiment_tag,
                seq_len_combinations=(
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    1310726,
                ),
            )

            # save failed combinations
            filename = f"{experiment_tag}_failed_combinations.json".replace("/", "_")
            with open(filename, "w") as json_file:
                json.dump(all_failed_combinations, json_file, indent=4)

            print(f"Saved {filename} to the current directory")
        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"Failed to run {yaml_name} with {recipe_type} with {e} and {tb}")
            continue

# TODO: fix decline last step for tokens per second
