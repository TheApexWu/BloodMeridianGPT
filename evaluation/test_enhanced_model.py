#!/usr/bin/env python3
"""
Test script for enhanced McCarthy GPT model with comprehensive validation.
"""

import torch
import numpy as np
import time
from model_refined import RefinedMcCarthyGPT, RefinedConfig
from training_refined import load_data, estimate_loss, get_batch

def test_model_creation():
    """Test that the enhanced model can be created successfully."""
    print("=== Testing Model Creation ===")
    try:
        config = RefinedConfig()
        model = RefinedMcCarthyGPT(config)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
        print(f"  Context window: {config.block_size}")
        print(f"  Layers: {config.n_layer}")
        print(f"  Attention heads: {config.n_head}")
        return model, config
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        return None, None

def test_forward_pass(model, config):
    """Test forward pass with random data."""
    print("\n=== Testing Forward Pass ===")
    try:
        # Create random input
        batch_size = 4
        seq_len = 128
        x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Forward pass without targets (no loss)
        logits, loss = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {logits.shape}")
        print(f"  Loss: {loss}")
        print(f"  Expected output shape: ({batch_size}, {seq_len}, {config.vocab_size})")
        
        # Validate shapes
        assert logits.shape == (batch_size, seq_len, config.vocab_size), "Wrong output shape"
        
        # Test forward pass with targets (with loss)
        targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        logits_with_loss, loss_with_loss = model(x, targets)
        
        print(f"  Loss with targets: {loss_with_loss.item():.4f}")
        assert loss_with_loss is not None, "Loss should not be None when targets provided"
        assert loss_with_loss.item() > 0, "Loss should be positive"
        
        return True
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False

def test_generation(model, config):
    """Test text generation capabilities."""
    print("\n=== Testing Generation ===")
    try:
        # Test generation
        start_token = torch.zeros((1, 1), dtype=torch.long)
        generated = model.generate(start_token, max_new_tokens=50, temperature=1.0)
        
        print(f"✓ Generation successful")
        print(f"  Generated shape: {generated.shape}")
        print(f"  Generated length: {generated.shape[1]}")
        print(f"  Sample tokens: {generated[0][:10].tolist()}")
        
        # Test different sampling parameters
        for temp in [0.5, 1.0, 1.5]:
            gen = model.generate(start_token, max_new_tokens=10, temperature=temp)
            print(f"  Temperature {temp}: {gen[0][:10].tolist()}")
        
        return True
    except Exception as e:
        print(f"✗ Generation failed: {e}")
        return False

def test_attention_mechanism(model, config):
    """Test that ALiBi attention is working correctly."""
    print("\n=== Testing ALiBi Attention ===")
    try:
        # Create a simple attention test
        batch_size = 2
        seq_len = 64
        x = torch.randn(batch_size, seq_len, config.n_embd)
        
        # Get attention block
        attention_block = model.blocks[0].attn
        
        # Test forward pass through attention
        output = attention_block(x)
        
        print(f"✓ ALiBi attention working")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  ALiBi slopes shape: {attention_block.alibi_slope.shape}")
        
        # Verify ALiBi slopes are applied
        assert hasattr(attention_block, 'alibi_slope'), "ALiBi slopes not found"
        assert attention_block.alibi_slope.shape[1] == config.n_head, "Wrong number of slopes"
        
        return True
    except Exception as e:
        print(f"✗ ALiBi attention test failed: {e}")
        return False

def test_swiglu_activation(model, config):
    """Test SwiGLU activation in feed-forward network."""
    print("\n=== Testing SwiGLU Activation ===")
    try:
        # Get feed-forward block
        ff_block = model.blocks[0].ff
        
        # Test with random input
        batch_size = 2
        seq_len = 32
        x = torch.randn(batch_size, seq_len, config.n_embd)
        
        output = ff_block(x)
        
        print(f"✓ SwiGLU activation working")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Hidden dimension: {ff_block.gate_proj.out_features}")
        
        # Verify shapes
        assert output.shape == x.shape, "Output shape mismatch"
        assert ff_block.gate_proj.out_features == ff_block.up_proj.out_features, "Gate and up projections should have same size"
        
        return True
    except Exception as e:
        print(f"✗ SwiGLU activation test failed: {e}")
        return False

def test_memory_efficiency(model, config):
    """Test memory usage and gradient accumulation."""
    print("\n=== Testing Memory Efficiency ===")
    try:
        # Test gradient accumulation simulation
        batch_size = 8
        seq_len = 256
        grad_accum_steps = 4
        
        # Simulate gradient accumulation
        total_loss = 0.0
        model.train()
        
        for step in range(grad_accum_steps):
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            logits, loss = model(x, targets)
            scaled_loss = loss / grad_accum_steps
            total_loss += scaled_loss.item()
            
            # Simulate backward pass (without actually calling backward)
            print(f"  Step {step + 1}: loss = {scaled_loss.item():.4f}")
        
        print(f"✓ Gradient accumulation simulation successful")
        print(f"  Total scaled loss: {total_loss:.4f}")
        print(f"  Effective batch size: {batch_size * grad_accum_steps}")
        
        return True
    except Exception as e:
        print(f"✗ Memory efficiency test failed: {e}")
        return False

def benchmark_model_performance(model, config):
    """Benchmark model performance and compare to original."""
    print("\n=== Performance Benchmark ===")
    try:
        # Test different sequence lengths
        seq_lengths = [256, 512, 768]
        batch_size = 4
        
        for seq_len in seq_lengths:
            if seq_len > config.block_size:
                print(f"  Skipping {seq_len} (exceeds block_size)")
                continue
                
            x = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            
            # Time forward pass
            start_time = time.time()
            logits, loss = model(x)
            forward_time = time.time() - start_time
            
            # Time generation
            start_time = time.time()
            generated = model.generate(x[:, :1], max_new_tokens=50)
            generation_time = time.time() - start_time
            
            print(f"  Sequence length {seq_len}:")
            print(f"    Forward pass: {forward_time:.3f}s")
            print(f"    Generation (50 tokens): {generation_time:.3f}s")
            print(f"    Memory estimate: ~{seq_len * batch_size * config.n_embd * 4 / 1024 / 1024:.1f}MB")
        
        return True
    except Exception as e:
        print(f"✗ Performance benchmark failed: {e}")
        return False

def test_sampling_strategies(model, config):
    """Test different sampling strategies for diversity."""
    print("\n=== Testing Sampling Strategies ===")
    try:
        start_token = torch.zeros((1, 1), dtype=torch.long)
        
        # Test different sampling strategies
        strategies = [
            {"temperature": 0.5, "top_k": None, "top_p": None, "name": "Low temp"},
            {"temperature": 1.0, "top_k": None, "top_p": None, "name": "Normal"},
            {"temperature": 1.5, "top_k": None, "top_p": None, "name": "High temp"},
            {"temperature": 1.0, "top_k": 20, "top_p": None, "name": "Top-k=20"},
            {"temperature": 1.0, "top_k": None, "top_p": 0.9, "name": "Top-p=0.9"},
            {"temperature": 1.0, "top_k": 50, "top_p": 0.9, "name": "Combined"},
        ]
        
        for strategy in strategies:
            generated = model.generate(
                start_token, 
                max_new_tokens=100,
                temperature=strategy["temperature"],
                top_k=strategy["top_k"],
                top_p=strategy["top_p"]
            )
            
            # Calculate diversity metrics
            tokens = generated[0].tolist()
            unique_tokens = len(set(tokens))
            total_tokens = len(tokens)
            diversity = unique_tokens / total_tokens
            
            print(f"  {strategy['name']}: diversity = {diversity:.3f}")
            print(f"    Sample: {tokens[:20]}")
        
        return True
    except Exception as e:
        print(f"✗ Sampling strategies test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide a summary."""
    print("Enhanced McCarthy GPT - Comprehensive Test Suite")
    print("=" * 60)
    
    # Test model creation
    model, config = test_model_creation()
    if model is None:
        print("\n Cannot proceed - model creation failed")
        return False
    
    # Run all tests
    tests = [
        ("Forward Pass", lambda: test_forward_pass(model, config)),
        ("Generation", lambda: test_generation(model, config)),
        ("ALiBi Attention", lambda: test_attention_mechanism(model, config)),
        ("SwiGLU Activation", lambda: test_swiglu_activation(model, config)),
        ("Memory Efficiency", lambda: test_memory_efficiency(model, config)),
        ("Performance Benchmark", lambda: benchmark_model_performance(model, config)),
        ("Sampling Strategies", lambda: test_sampling_strategies(model, config)),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print(" All tests passed! Enhanced model is ready for training.")
    else:
        print("  Some tests failed. Check the implementation.")
    
    return passed == len(results)

if __name__ == '__main__':
    success = run_comprehensive_test()
    exit(0 if success else 1)