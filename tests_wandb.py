# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import pandas as pd
import wandb


def fetch_and_process_data(project, experiment_tag):
    api = wandb.Api()
    runs = api.runs(f"{project}", filters={"tags": {"$all": [experiment_tag]}})

    data = []
    for run in runs:
        history = run.history(samples=10000)
        if len(history) == 0:
            continue

        peak_memory_alloc = history["peak_memory_alloc"]
        tokens_per_second_per_gpu = history["tokens_per_second_per_gpu"]

        # runtime
        run_time = history["_runtime"]
        total_run_time = run_time.iloc[-1]
        time_to_first_batch = run_time.iloc[0]

        # max_peak_memory_alloc, mean_tokens_per_second
        if len(peak_memory_alloc) > 1:
            # Dropping first and last batch
            max_peak_memory_alloc = max(peak_memory_alloc.iloc[1:-1])
            mean_tokens_per_second = sum(tokens_per_second_per_gpu.iloc[1:-1]) / len(
                tokens_per_second_per_gpu.iloc[1:-1]
            )
        else:
            max_peak_memory_alloc = peak_memory_alloc[0]
            mean_tokens_per_second = tokens_per_second_per_gpu[0]

        data.append(
            {
                "run_name": run.name,
                "tags": tuple([tag for tag in run.tags if tag != experiment_tag]),
                "description": tuple(run.config.get("run_tag", [])),
                "batch_size": run.config.get("batch_size", None),
                "max_seq_len": run.config.get("dataset", {}).get("max_seq_len", None),
                "num_steps": len(history),
                "max_peak_memory_alloc": round(max_peak_memory_alloc, 2),
                "mean_tokens_per_second_per_gpu": round(mean_tokens_per_second, 2),
                "time_to_first_batch": round(time_to_first_batch, 2),
                "total_run_time": round(total_run_time, 2),
            }
        )

    # Creating DataFrame from list of dictionaries
    df = pd.DataFrame(data)
    return df


def create_graphs(df, filter_tag):
    # Define markers and colors for up to five different configurations
    config_styles = {
        "default_config": {"marker": "*", "colors": ["black", "blue"]},
        "memory_optimized_config": {"marker": "^", "colors": ["red", "purple"]},
        # Add more configurations as needed
    }

    # Filter DataFrame for graphs
    df_graphs = df[df["tags"].apply(lambda tags: filter_tag not in tags)]

    # Plotting max_peak_memory_alloc Vs max_seq_len
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    for (tags, bsz), group in df_graphs.groupby(["tags", "batch_size"]):
        config = tags[0] if tags else "default"
        style = config_styles.get(
            config, {"marker": "o", "colors": ["grey"]}
        )  # Default style
        color = style["colors"][
            int(bsz) % len(style["colors"])
        ]  # Cycle through colors if more batch sizes
        marker = style["marker"]
        group = group.sort_values(
            by="max_seq_len"
        )  # Ensure lines connect in order of sequence length
        ax.plot(
            group["max_seq_len"],
            group["max_peak_memory_alloc"],
            label=f"{tags}_{bsz=}",
            marker=marker,
            color=color,
            linestyle="-",
        )

    plt.xlabel("Max Sequence Length")
    plt.ylabel("Max Peak Memory Allocation")
    plt.title("Max Peak Memory Allocation vs Max Sequence Length")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig("max_peak_memory_alloc_vs_max_seq_len.png")
    plt.close()

    # Plotting mean_tokens_per_second_per_gpu Vs max_seq_len
    plt.figure(figsize=(10, 5))
    ax = plt.gca()
    for (tags, bsz), group in df_graphs.groupby(["tags", "batch_size"]):
        config = tags[0] if tags else "default"
        style = config_styles.get(config, {"marker": "o", "colors": ["grey"]})
        color = style["colors"][int(bsz) % len(style["colors"])]
        marker = style["marker"]
        group = group.sort_values(by="max_seq_len")
        ax.plot(
            group["max_seq_len"],
            group["mean_tokens_per_second_per_gpu"],
            label=f"{tags}_{bsz=}",
            marker=marker,
            color=color,
            linestyle="-",
        )

    plt.xlabel("Max Sequence Length")
    plt.ylabel("Mean Tokens per Second per GPU")
    plt.title("Mean Tokens per Second per GPU vs Max Sequence Length")
    plt.grid(True)
    plt.legend(loc="upper right")
    plt.savefig("mean_tokens_per_second_per_gpu_vs_max_seq_len.png")
    plt.close()


def create_comparison_table(df, config_tags):
    # Extract relevant configurations
    relevant_tags = list(config_tags.values()) + [config_tags["one_by_one_config_tag"]]

    # Filter DataFrame for relevant configurations
    df_filtered = df[
        df["tags"].apply(lambda tags: any(tag in tags for tag in relevant_tags))
    ]

    # Get unique (max_seq_len, batch_size) tuples from "one_by_one_config"
    one_by_one_conditions = df_filtered[
        df_filtered["tags"].apply(
            lambda tags: config_tags["one_by_one_config_tag"] in tags
        )
    ][["max_seq_len", "batch_size"]].drop_duplicates()

    # Further filter to include only those entries with matching max_seq_len and batch_size
    df_filtered = df_filtered.merge(
        one_by_one_conditions, on=["max_seq_len", "batch_size"]
    )

    # Calculate percentage differences
    df_comparison = pd.DataFrame()
    for (max_seq_len, batch_size), group in df_filtered.groupby(
        ["max_seq_len", "batch_size"]
    ):
        baseline = group[
            group["tags"].apply(lambda tags: config_tags["default_config_tag"] in tags)
        ]
        rows_to_add = []
        for _, row in group.iterrows():
            if baseline.empty or row["tags"] == baseline.iloc[0]["tags"]:
                percent_diff_memory = 0
                percent_diff_tokens = 0
            else:
                percent_diff_memory = (
                    (
                        row["max_peak_memory_alloc"]
                        - baseline.iloc[0]["max_peak_memory_alloc"]
                    )
                    / baseline.iloc[0]["max_peak_memory_alloc"]
                ) * 100
                percent_diff_tokens = (
                    (
                        row["mean_tokens_per_second_per_gpu"]
                        - baseline.iloc[0]["mean_tokens_per_second_per_gpu"]
                    )
                    / baseline.iloc[0]["mean_tokens_per_second_per_gpu"]
                ) * 100

            rows_to_add.append(
                {
                    "tags": row["tags"][0],
                    "description": "__".join(row["description"]),
                    "batch_size": row["batch_size"],
                    "max_seq_len": row["max_seq_len"],
                    "max_peak_memory_alloc": row["max_peak_memory_alloc"],
                    "mean_tokens_per_second_per_gpu": row[
                        "mean_tokens_per_second_per_gpu"
                    ],
                    "pct_diff_memory_vs_default": round(percent_diff_memory, 2),
                    "pct_diff_tokens_vs_default": round(percent_diff_tokens, 2),
                }
            )

        df_comparison = pd.concat(
            [df_comparison, pd.DataFrame(rows_to_add)], ignore_index=True
        )

    # Sort the DataFrame
    # Define a custom sort order for configurations
    sort_priority = {
        config_tags["default_config_tag"]: 1,
        config_tags["memory_optimized_config_tag"]: 2,
        config_tags["speed_optimized_config_tag"]: 3,
    }
    df_comparison["sort_priority"] = df_comparison["tags"].apply(
        lambda x: sort_priority.get(x, 4)
    )
    df_comparison.sort_values(
        by=[
            "max_seq_len",
            "batch_size",
            "sort_priority",
            "pct_diff_memory_vs_default",
            "pct_diff_tokens_vs_default",
        ],
        ascending=[True, True, True, True, False],
        inplace=True,
    )
    df_comparison.drop(columns=["sort_priority"], inplace=True)

    # Save to CSV
    df_comparison.to_csv("comparison_table.csv", index=False)
    return df_comparison


if __name__ == "__main__":
    # Your existing code here...
    experiment_tag = "llama3_1/8B_lora_single_device_2024_08_02_04_09"
    # Assuming DEFAULT_FLAGS and experiment_tag are defined
    project = "recipe_profiling"  # DEFAULT_FLAGS["metric_logger.project"]["values"][0]
    entity = None  # Assuming entity is not needed or is set elsewhere

    # Call the function to fetch and process data
    df = fetch_and_process_data(project, experiment_tag)
    df.to_csv("experiment_results.csv", index=False)

    # Create graphs and table
    filter_tag = "one_by_one_config"
    create_graphs(df, filter_tag)

    # Define configuration tags
    config_tags = {
        "default_config_tag": "default_config",
        "memory_optimized_config_tag": "memory_optimized_config",
        "speed_optimized_config_tag": "speed_optimized_config",
        "one_by_one_config_tag": "one_by_one_config",
    }
    # Create comparison table
    comparison_df = create_comparison_table(df, config_tags)
