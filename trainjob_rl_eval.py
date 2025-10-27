"""
Kubeflow TrainJob for nanochat RL model evaluation.

This script creates a TrainJob that evaluates a trained RL model on full GSM8K test set.

Usage:
    python trainjob_rl_eval.py
"""

from kubeflow.trainer import TrainerClient, CustomTrainer


def nanochat_rl_evaluation():
    """Evaluation function for nanochat RL model that will run in the TrainJob"""
    
    import os
    import sys
    import torch
    import torch.distributed as dist
    
    # Set environment variables
    USER_WORKSPACE = os.getenv("USER_WORKSPACE", "/data/workspace/kimberly")
    
    # Add nanochat to Python path (nanochat code is already in PVC)
    nanochat_path = os.path.join(USER_WORKSPACE, "nanochat")
    if nanochat_path not in sys.path:
        sys.path.insert(0, nanochat_path)
    print(f"Added nanochat to sys.path: {nanochat_path}")
    
    # Import nanochat modules
    from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir
    from nanochat.checkpoint_manager import load_model
    from nanochat.engine import Engine
    from tasks.gsm8k import GSM8K
    os.environ["NANOCHAT_BASE_DIR"] = f"{USER_WORKSPACE}/.cache/nanochat"
    os.environ["HF_HOME"] = f"{USER_WORKSPACE}/.cache/huggingface"
    os.environ["UV_CACHE_DIR"] = f"{USER_WORKSPACE}/.cache/uv"
    os.environ["OMP_NUM_THREADS"] = "1"

    print(f"USER_WORKSPACE: {USER_WORKSPACE}")
    print(f"NANOCHAT_BASE_DIR: {os.environ['NANOCHAT_BASE_DIR']}")
    
    # Evaluation configuration
    source = "rl"  # Load RL checkpoint
    dtype_str = "bfloat16"
    device_batch_size = 1  # Pass@1 evaluation (standard)
    max_new_tokens = 512  # Match chat_eval.py default (GSM8K needs longer answers)
    temperature = 0.0  # Greedy decoding (standard evaluation)
    top_k = 50
    
    # Init compute/precision
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init()
    master_process = ddp_rank == 0
    dtype = torch.float32 if dtype_str == 'float32' else torch.bfloat16
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=dtype)
    
    print(f"DDP initialized: world_size={ddp_world_size}, rank={ddp_rank}, local_rank={ddp_local_rank}")
    
    # Load RL model and tokenizer
    print0(f"Loading RL model from source: {source}")
    model, tokenizer, meta = load_model(source, device, phase="eval")
    engine = Engine(model, tokenizer)
    
    # Load GSM8K test set
    val_task = GSM8K(subset="main", split="test")
    print0(f"Test set size: {len(val_task)}")
    
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
    
    # Run evaluation on full test set
    print0("\n" + "="*80)
    print0("Running standard evaluation on full GSM8K test set...")
    print0(f"Evaluation mode: Pass@1 (greedy decoding)")
    print0(f"Temperature: {temperature}")
    print0(f"Test set size: {len(val_task)}")
    print0("="*80 + "\n")
    
    model.eval()
    passk_final = torch.zeros(device_batch_size, device=device)
    
    with autocast_ctx:
        records_iter = run_gsm8k_eval(
            val_task, tokenizer, engine,
            num_samples=device_batch_size,
            max_examples=None,  # Full test set!
            max_completion_tokens=max_new_tokens,
            temperature=temperature
        )
        records = list(records_iter)
    
    # Calculate Pass@k on full test set
    for k in range(1, device_batch_size + 1):
        passk_final[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
    
    num_records_final = torch.tensor(len(records), dtype=torch.long, device=device)
    if ddp:
        dist.all_reduce(num_records_final, op=dist.ReduceOp.SUM)
        dist.all_reduce(passk_final, op=dist.ReduceOp.SUM)
    
    passk_final = passk_final / num_records_final.item()
    pass1_accuracy = passk_final[0].item()
    
    print0(f"\n🎯 Final Evaluation Results")
    print0(f"Pass@1 Accuracy: {pass1_accuracy:.4f} ({pass1_accuracy*100:.2f}%)")
    
    # Cleanup
    if ddp:
        dist.barrier()
    
    if master_process:
        print0("\n" + "="*80)
        print0("RL Model Evaluation completed successfully!")
        print0("="*80)
    
    # Log to report
    from nanochat.report import get_report
    
    eval_config = {
        "source": source,
        "dtype": dtype_str,
        "device_batch_size": device_batch_size,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
    }
    
    get_report().log(section="Chat RL Model Evaluation (Kubeflow)", data=[
        eval_config,  # Evaluation config
        { # Evaluation results
            "Test set": "GSM8K",
            "Test set size": len(val_task),
            "Model checkpoint": meta.get("checkpoint_path", "chatrl latest"),
            "Infrastructure": "2x H100 GPUs via Kubeflow",
        },
        { # Final Pass@1 result (standard evaluation)
            "Pass@1 Accuracy": f"{pass1_accuracy:.4f}",
            "Pass@1 Percentage": f"{pass1_accuracy*100:.2f}%",
        }
    ])
    
    compute_cleanup()


if __name__ == "__main__":
    # Kubeflow configuration
    client = TrainerClient(namespace="yunjae-park-kf-profile")

    for r in client.list_runtimes():
        print(f"Runtime: {r.name}")

    # Create the TrainJob for evaluation
    job_id = client.train(
        runtime_ref="torch-distributed",
        trainer=CustomTrainer(
            func=nanochat_rl_evaluation,
            num_nodes=1,
            resources_per_node={
                "gpu": 2,  # 2 GPUs for faster evaluation
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
            ],
        ),
    )
    
    print(f"\n✅ Evaluation TrainJob created successfully!")
    print(f"Job ID: {job_id}")
    print(f"\nTo monitor the job:")
    print(f"  kubectl get trainjob {job_id} -n yunjae-park-kf-profile")
    print(f"  kubectl logs -f job/{job_id}-node-0 -n yunjae-park-kf-profile")

