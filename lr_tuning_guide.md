# A Guide to Finding the Optimal Learning Rate

Finding the right learning rate (LR) is crucial for successful model training. As you've noted, it's influenced by many hyperparameters like batch size, training duration, and parallelism strategy. This guide outlines several methods to find a good LR within a fixed budget.

## 1. Understanding Your Effective Batch Size

Before tuning the LR, it's important to know your *effective batch size*. This is the total number of samples the model sees before the weights are updated. In `full_finetune_distributed.py`, it's calculated as:

`Effective Batch Size = batch_size * a * b`

where:
- `batch_size` is the per-device value from your config.
- `a` is the number of GPUs used for data parallelism (`dp_degree`).
- `b` is the `gradient_accumulation_steps`.

Changes to any of these will change your effective batch size and likely require you to adjust your LR.

## 2. Strategies for Finding a Good Learning Rate

### Strategy 1: Start with a Good Default

For well-known models and tasks, there are often community-vetted LR values. For example, the `llama3_3/70B_full.yaml` config suggests `lr: 2e-5`. This is an excellent starting point. For many finetuning tasks on LLMs, LRs between `1e-5` and `5e-5` with an AdamW optimizer are common.

**Action**: Always try the default or a community-recommended LR first.

### Strategy 2: Use Heuristics (The Linear Scaling Rule)

A popular heuristic is the **Linear Scaling Rule**: "When the minibatch size is multiplied by k, multiply the learning rate by k (or sqrt(k))".

- If you double your effective batch size, you can try doubling your LR (or increasing it by ~1.4x).

This works because larger batches provide a more accurate estimate of the gradient, allowing for larger, more confident steps.

**Action**: If you change your batch size, use this rule to find a new starting LR.

### Strategy 3: The LR Range Test (Most Recommended for a Fixed Budget)

This is a systematic way to find a good LR range in a single, short training run.

**How it works:**
1.  **Setup**: Pick a very small starting LR (e.g., `1e-8`) and a large maximum LR (e.g., `1e-1`).
2.  **Run**: Train the model for one epoch (or even just a few hundred steps). At each training step, use a slightly higher LR than the previous step, moving from your start LR to your max LR.
3.  **Plot**: Log the loss at each step against the LR used for that step. Plot loss vs. LR (with LR on a log scale).
4.  **Analyze**: The plot will typically show the loss decreasing, then flattening out, and finally increasing sharply.

![LR Range Test Plot Example](https://raw.githubusercontent.com/davidtvs/pytorch-lr-finder/master/examples/lr_finder_result.png)
*Image source: pytorch-lr-finder repository*

**How to pick the LR from the plot:**
- **Best guess**: The best LR is usually at the point of the steepest decline, right before the loss starts to flatten.
- **Rule of thumb**: Pick a value that is one order of magnitude lower than the LR where the loss is at its minimum (before it shoots up). In the example plot, the minimum is around `1e-2`, so a good LR to try would be `1e-3`.

**Action**: Implement a short training script that sweeps the LR and plots the loss to find your optimal LR. This is the most efficient way to explore the LR space.

### Strategy 4: Always Use a Learning Rate Scheduler

A constant learning rate is rarely optimal throughout training. A learning rate scheduler dynamically adjusts the LR. For finetuning, a common and effective schedule is **cosine decay with warmup**.

1.  **Warmup**: For the first N steps, the LR increases linearly from 0 to your peak LR (the one you found with the LR range test). This prevents instabilities at the start of training.
2.  **Decay**: After warmup, the LR smoothly decreases following a cosine curve.

**Action**: Add an LR scheduler to your configuration. `torchtune` supports this easily. In your YAML config:

```yaml
# In your config file (e.g., 70B_full.yaml)

optimizer:
  _component_: torch.optim.AdamW
  lr: 2e-5 # Your peak LR found from the range test
  fused: False

lr_scheduler:
  _component_: torchtune.training.get_cosine_schedule_with_warmup
  # A common value is 5-10% of total training steps
  num_warmup_steps: 150

# ... rest of config
```

By combining these techniques, you can find a near-optimal learning rate efficiently and improve your model's performance. 