"""
Kubeflow TrainJob for nanochat Reinforcement Learning (RL) training.

This script creates a TrainJob that runs chat_rl training on GSM8K using Kubeflow Trainer.

Usage:
    python trainjob_rl.py
"""

from kubeflow.trainer import TrainerClient, CustomTrainer
from kubeflow.trainer.backends.kubernetes.types import KubernetesBackendConfig


def nanochat_rl_training():
    """Main training function for nanochat RL that will run in the TrainJob"""
    
    import os
    import sys
    import itertools
    import torch
    import torch.distributed as dist
    
    # Set environment variables
    USER_WORKSPACE = os.getenv("USER_WORKSPACE", "/data/workspace/kimberly")
    os.environ["WANDB_API_KEY"] = "dc07c5d951fc5c844b15752232fde38909adec05"
    os.environ["WANDB_RUN"] = "nanochat_rl_kubeflow"
    
    # Add nanochat to Python path (nanochat code is already in PVC)
    nanochat_path = os.path.join(USER_WORKSPACE, "nanochat")
    if nanochat_path not in sys.path:
        sys.path.insert(0, nanochat_path)
    print(f"Added nanochat to sys.path: {nanochat_path}")
    
    # Import nanochat modules
    from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb
    from nanochat.checkpoint_manager import save_checkpoint, load_model
    from nanochat.engine import Engine
    from tasks.gsm8k import GSM8K
    os.environ["NANOCHAT_BASE_DIR"] = f"{USER_WORKSPACE}/.cache/nanochat"
    os.environ["HF_HOME"] = f"{USER_WORKSPACE}/.cache/huggingface"
    os.environ["UV_CACHE_DIR"] = f"{USER_WORKSPACE}/.cache/uv"
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"USER_WORKSPACE: {USER_WORKSPACE}")
    print(f"NANOCHAT_BASE_DIR: {os.environ['NANOCHAT_BASE_DIR']}")
    
    # RL hyperparameters
    run = os.getenv("WANDB_RUN", "nanochat_rl_kubeflow")
    source = "sft"  # mid|sft
    dtype_str = "bfloat16"
    
    # For 2 GPU: adjusted for faster testing with lower batch size
    device_batch_size = 8  # Per GPU batch size
    examples_per_step = 16  # Total examples per step (divided across 2 GPUs)
    num_samples = 16  # Number of samples per example
    max_new_tokens = 256
    temperature = 1.0
    top_k = 50
    
    
    unembedding_lr = 0.004
    embedding_lr = 0.2
    matrix_lr = 0.02
    weight_decay = 0.0
    init_lr_frac = 0.05
    num_epochs = 1  
    save_every = 60
    eval_every = 60
    eval_examples = 400
    
    user_config = {
        "run": run,
        "source": source,
        "dtype": dtype_str,
        "device_batch_size": device_batch_size,
        "examples_per_step": examples_per_step,
        "num_samples": num_samples,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "unembedding_lr": unembedding_lr,
        "embedding_lr": embedding_lr,
        "matrix_lr": matrix_lr,
        "weight_decay": weight_decay,
        "init_lr_frac": init_lr_frac,
        "num_epochs": num_epochs,
        "save_every": save_every,
        "eval_every": eval_every,
        "eval_examples": eval_examples,
    }
    
    # Init compute/precision
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    master_process = ddp_rank == 0
    dtype = torch.float32 if dtype_str == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    
    print(f"DDP initialized: world_size={ddp_world_size}, rank={ddp_rank}, local_rank={ddp_local_rank}")
    
    # WandB logging (optional)
    try:
        import wandb
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key and run != "dummy":
            wandb.login(key=wandb_api_key, relogin=True)
            wandb_run = wandb.init(project="nanochat-rl", name=run, config=user_config) if master_process else DummyWandb()
        else:
            print("WANDB_API_KEY not set or run='dummy', using DummyWandb")
            wandb_run = DummyWandb()
    except ImportError:
        print("WandB not available, using DummyWandb")
        wandb_run = DummyWandb()
    
    # Load model and tokenizer
    print0(f"Loading model from source: {source}")
    model, tokenizer, meta = load_model(source, device, phase="eval")
    engine = Engine(model, tokenizer)
    
    # Load GSM8K tasks
    train_task = GSM8K(subset="main", split="train")
    val_task = GSM8K(subset="main", split="test")
    num_steps = (len(train_task) // examples_per_step) * num_epochs
    print0(f"Calculated number of steps: {num_steps}")
    
    # Rollout generator function
    @torch.no_grad()
    def get_batch():
        assistant_end = tokenizer.encode_special("<|assistant_end|>")
        rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
        
        for example_idx in itertools.cycle(rank_indices):
            conversation = train_task[example_idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            
            # Generate samples
            model.eval()
            generated_token_sequences = []
            masks = []
            num_sampling_steps = num_samples // device_batch_size
            
            for sampling_step in range(num_sampling_steps):
                seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF  # step from outer scope
                with autocast_ctx:
                    generated_token_sequences_batch, masks_batch = engine.generate_batch(
                        tokens,
                        num_samples=device_batch_size,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        top_k=top_k,
                        seed=seed,
                    )
                generated_token_sequences.extend(generated_token_sequences_batch)
                masks.extend(masks_batch)
            
            # Calculate rewards
            rewards = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                reward = train_task.reward(conversation, generated_text)
                rewards.append(reward)
            
            # Pad sequences
            max_length = max(len(seq) for seq in generated_token_sequences)
            padded_generated_token_sequences = [
                seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences
            ]
            padded_masks = [mask + [0] * (max_length - len(mask)) for mask in masks]
            
            # Create tensors
            ids = torch.tensor(padded_generated_token_sequences, dtype=torch.long, device=device)
            mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
            inputs = ids[:, :-1]
            targets = ids[:, 1:].clone()
            targets[mask_ids[:, 1:] == 0] = -1
            rewards = torch.tensor(rewards, dtype=torch.float, device=device)
            
            # Calculate advantages
            mu = rewards.mean()
            advantages = rewards - mu
            
            yield generated_token_sequences, inputs, targets, rewards, advantages
    
    # Evaluation function
    def run_gsm8k_eval(task, tokenizer, engine, max_examples=None, num_samples=1,
                       max_completion_tokens=256, temperature=0.0, top_k=50):
        max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
        for idx in range(ddp_rank, max_examples, ddp_world_size):
            conversation = task[idx]
            tokens = tokenizer.render_for_completion(conversation)
            prefix_length = len(tokens)
            
            generated_token_sequences, masks = engine.generate_batch(
                tokens,
                num_samples=num_samples,
                max_tokens=max_completion_tokens,
                temperature=temperature,
                top_k=top_k
            )
            
            outcomes = []
            for sample_tokens in generated_token_sequences:
                generated_tokens = sample_tokens[prefix_length:]
                generated_text = tokenizer.decode(generated_tokens)
                is_correct = task.evaluate(conversation, generated_text)
                outcomes.append({"is_correct": is_correct})
            
            yield {"idx": idx, "outcomes": outcomes}
    
    # Initialize optimizer
    optimizers = model.setup_optimizers(
        unembedding_lr=unembedding_lr,
        embedding_lr=embedding_lr,
        matrix_lr=matrix_lr,
        weight_decay=weight_decay,
    )
    
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["lr"] * init_lr_frac
            group["initial_lr"] = group["lr"]
    
    def get_lr_multiplier(it):
        return 1.0 - it / num_steps
    
    print0(f"Total sequences per step: {examples_per_step * num_samples}")
    assert examples_per_step % ddp_world_size == 0
    examples_per_rank = examples_per_step // ddp_world_size
    print0(f"Calculated examples per rank: {examples_per_rank}")
    
    # Training loop
    print0("\n" + "="*80)
    print0("Starting RL Training")
    print0(f"Total steps: {num_steps}")
    print0(f"Device batch size: {device_batch_size}")
    print0(f"Num samples per example: {num_samples}")
    print0(f"Examples per step: {examples_per_step}")
    print0("="*80 + "\n")
    
    # Create batch iterator once before the loop
    batch_iterator = get_batch()
    
    for step in range(num_steps):
        
        # Evaluation
        if step % eval_every == 0:
            model.eval()
            passk = torch.zeros(device_batch_size, device=device)
            with autocast_ctx:
                records_iter = run_gsm8k_eval(
                    val_task, tokenizer, engine,
                    num_samples=device_batch_size,
                    max_examples=eval_examples,
                    temperature=1.0
                )
                records = list(records_iter)
            
            for k in range(1, device_batch_size + 1):
                passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
            
            num_records = torch.tensor(len(records), dtype=torch.long, device=device)
            if ddp:
                dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
                dist.all_reduce(passk, op=dist.ReduceOp.SUM)
            
            passk = passk / num_records.item()
            print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, device_batch_size + 1)]
            print0(f"Step {step} | {', '.join(print_passk)}")
            
            log_passk = {f"pass@{k}": passk[k - 1].item() for k in range(1, device_batch_size + 1)}
            wandb_run.log({"step": step, **log_passk})
        
        # Training step
        rewards_list = []
        sequence_lengths = []
        
        for example_step in range(examples_per_rank):
            sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
            
            model.train()
            assert inputs_all.size(0) % device_batch_size == 0
            num_passes = inputs_all.size(0) // device_batch_size
            
            for pass_idx in range(num_passes):
                b0, b1 = pass_idx * device_batch_size, (pass_idx + 1) * device_batch_size
                inputs = inputs_all[b0:b1]
                targets = targets_all[b0:b1]
                rewards = rewards_all[b0:b1]
                advantages = advantages_all[b0:b1]
                
                with autocast_ctx:
                    logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
                
                pg_obj = (logp * advantages.unsqueeze(-1)).sum()
                num_valid = (targets >= 0).sum().clamp(min=1)
                pg_obj = pg_obj / (num_valid * num_passes * examples_per_rank)
                loss = -pg_obj
                loss.backward()
                
                print0(f"Step {step}/{num_steps} | Example {example_step} | Pass {pass_idx} | loss: {loss.item():.6f} | Avg reward: {rewards.mean().item():.4f}")
            
            rewards_list.append(rewards_all.mean().item())
            sequence_lengths.extend(len(seq) for seq in sequences_all)
        
        # Logging
        mean_reward = sum(rewards_list) / len(rewards_list)
        mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
        
        if ddp:
            mean_reward_tensor = torch.tensor(mean_reward, dtype=torch.float, device=device)
            mean_sequence_length_tensor = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
            dist.all_reduce(mean_reward_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(mean_sequence_length_tensor, op=dist.ReduceOp.AVG)
            mean_reward = mean_reward_tensor.item()
            mean_sequence_length = mean_sequence_length_tensor.item()
        
        print0(f"Step {step}/{num_steps} | Avg reward: {mean_reward:.4f} | Avg seq len: {mean_sequence_length:.2f}")
        wandb_run.log({
            "step": step,
            "reward": mean_reward,
            "sequence_length": mean_sequence_length,
        })
        
        # Update model
        lrm = get_lr_multiplier(step)
        for opt in optimizers:
            for group in opt.param_groups:
                group["lr"] = group["initial_lr"] * lrm
        for opt in optimizers:
            opt.step()
        model.zero_grad(set_to_none=True)
        
        wandb_run.log({"step": step, "lrm": lrm})
        
        # Save checkpoint
        if master_process and ((step > 0 and step % save_every == 0) or step == num_steps - 1):
            base_dir = get_base_dir()
            depth = model.config.n_layer
            model_tag = f"d{depth}"
            checkpoint_dir = os.path.join(base_dir, "chatrl_checkpoints", model_tag)
            model_config_kwargs = model.config.__dict__
            
            save_checkpoint(
                checkpoint_dir,
                step,
                model.state_dict(),
                None,
                {"model_config": model_config_kwargs}
            )
            print0(f"✅ Saved model checkpoint to {checkpoint_dir}")
    
    # Final evaluation on full test set (Pass@1 with greedy decoding)
    if master_process:
        print0("\n" + "="*80)
        print0("Running final evaluation on full GSM8K test set...")
        print0("Evaluation mode: Pass@1 (greedy decoding)")
        print0("="*80 + "\n")
    
    model.eval()
    pass1_final = torch.zeros(1, device=device)
    with autocast_ctx:
        records_iter = run_gsm8k_eval(
            val_task, tokenizer, engine,
            num_samples=1,  # Pass@1 only
            max_examples=None,  # Full test set!
            max_completion_tokens=512,  # Match chat_eval.py default
            temperature=0.0  # Greedy decoding
        )
        records = list(records_iter)
    
    # Calculate Pass@1 on full test set
    pass1_final[0] = sum(any(o["is_correct"] for o in r["outcomes"][:1]) for r in records)
    
    num_records_final = torch.tensor(len(records), dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(num_records_final, op=dist.ReduceOp.SUM)
        dist.all_reduce(pass1_final, op=dist.ReduceOp.SUM)
    
    pass1_final = pass1_final / num_records_final.item()
    pass1_accuracy = pass1_final[0].item()
    
    print0(f"\n🎯 Final Evaluation Results")
    print0(f"Pass@1 Accuracy: {pass1_accuracy:.4f} ({pass1_accuracy*100:.2f}%)")
    wandb_run.log({"step": num_steps, "final_pass@1": pass1_accuracy})
    
    # Cleanup
    if ddp:
        dist.barrier()
    
    if master_process:
        print0("\n" + "="*80)
        print0("RL Training completed successfully!")
        print0("="*80)
    
    # Log to report
    from nanochat.report import get_report
    get_report().log(section="Chat RL (Kubeflow TrainJob)", data=[
        user_config, # CLI args
        { # stats about the training setup
            "Number of training steps": num_steps,
            "Training examples": len(train_task),
            "Evaluation examples": eval_examples,
            "DDP world size": ddp_world_size,
            "Total sequences per step": examples_per_step * num_samples,
        },
        { # stats about training outcomes
            "Final checkpoint": f"model_{step:06d}.pt",
            "Training method": "Policy Gradient (REINFORCE)",
            "Initial model": f"{source} checkpoint",
            "Final model type": "RL-trained model",
            "Infrastructure": "2x H100 GPUs via Kubeflow",
            "Total training time": "~3 hours",
        },
        { # Final evaluation results on full test set
            "Test set": "GSM8K",
            "Test set size": len(val_task),
            "Pass@1 Accuracy": f"{pass1_accuracy:.4f}",
            "Pass@1 Percentage": f"{pass1_accuracy*100:.2f}%",
        }
    ])
    
    wandb_run.finish()
    compute_cleanup()


if __name__ == "__main__":
    # Kubeflow configuration
    # backend_config = KubernetesBackendConfig(namespace="yunjae-park-kf-profile")
    # client = TrainerClient(backend_config=backend_config)
    client = TrainerClient(namespace="yunjae-park-kf-profile")

    for r in client.list_runtimes():
        print(f"Runtime: {r.name}")

    # Create the TrainJob with FSDP support.
    job_id = client.train(
        # runtime=client.get_runtime("torch-distributed"),
        runtime_ref="torch-distributed",
        trainer=CustomTrainer(
            func=nanochat_rl_training,
            num_nodes=1,
            resources_per_node={
                "gpu": 2,  # 2 GPUs for faster training (~20 hours instead of ~80)
                "cpu": 8,
                "memory": "64Gi"
            },
            packages_to_install=[
                # Core PyTorch and ML packages
                "torch>=2.1.0",
                "numpy>=1.26.4",
                
                # nanochat dependencies (from pyproject.toml)
                "datasets>=4.0.0",
                "tokenizers>=0.22.0",
                "tiktoken>=0.11.0",
                "psutil>=7.1.0",
                "regex>=2024.9.1",
                
                # Optional: WandB for logging
                "wandb>=0.21.3",
            ],
        ),
    )
    
    print(f"\n✅ TrainJob created successfully!")
    print(f"Job ID: {job_id}")
    print(f"\nTo monitor the job:")
    print(f"  kubectl get trainjob {job_id} -n yunjae-park-kf-profile")
    print(f"  kubectl logs -f job/{job_id}-node-0 -n yunjae-park-kf-profile")

