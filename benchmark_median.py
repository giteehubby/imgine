#!/usr/bin/env python3

import torch
import time
from media import media_conv, media_conv_alternative, media_conv_original

def benchmark_median_filters():
    """比较不同中值滤波实现的性能"""
    
    print("=== 中值滤波性能测试 ===")
    
    # 创建测试图像
    test_sizes = [
        (3, 100, 100),   # 小图像
        (3, 200, 200),   # 中等图像
        (3, 400, 400),   # 大图像
    ]
    
    for channels, height, width in test_sizes:
        print(f"\n--- 测试图像尺寸: {channels}x{height}x{width} ---")
        
        # 创建随机测试图像
        test_img = torch.randint(0, 256, (channels, height, width), dtype=torch.uint8).float()
        
        # 测试原始实现
        print("测试原始实现...")
        start_time = time.time()
        try:
            result_original = media_conv_original(test_img.clone())
            original_time = time.time() - start_time
            print(f"  原始实现耗时: {original_time:.4f}秒")
        except Exception as e:
            print(f"  原始实现失败: {e}")
            original_time = float('inf')
        
        # 测试优化实现
        print("测试优化实现...")
        start_time = time.time()
        try:
            result_optimized = media_conv(test_img.clone())
            optimized_time = time.time() - start_time
            print(f"  优化实现耗时: {optimized_time:.4f}秒")
        except Exception as e:
            print(f"  优化实现失败: {e}")
            optimized_time = float('inf')
        
        # 测试替代实现
        print("测试替代实现...")
        start_time = time.time()
        try:
            result_alternative = media_conv_alternative(test_img.clone())
            alternative_time = time.time() - start_time
            print(f"  替代实现耗时: {alternative_time:.4f}秒")
        except Exception as e:
            print(f"  替代实现失败: {e}")
            alternative_time = float('inf')
        
        # 计算加速比
        if original_time != float('inf') and optimized_time != float('inf'):
            speedup = original_time / optimized_time
            print(f"  优化实现加速比: {speedup:.2f}x")
        
        if original_time != float('inf') and alternative_time != float('inf'):
            speedup_alt = original_time / alternative_time
            print(f"  替代实现加速比: {speedup_alt:.2f}x")
        
        # 验证结果一致性（如果都成功的话）
        if (original_time != float('inf') and optimized_time != float('inf') and 
            alternative_time != float('inf')):
            try:
                # 检查结果是否相似（允许小的数值误差）
                diff_opt = torch.abs(result_original - result_optimized).max()
                diff_alt = torch.abs(result_original - result_alternative).max()
                print(f"  结果差异检查:")
                print(f"    优化实现最大差异: {diff_opt:.6f}")
                print(f"    替代实现最大差异: {diff_alt:.6f}")
                
                if diff_opt < 1e-6 and diff_alt < 1e-6:
                    print("  ✅ 所有实现结果一致")
                else:
                    print("  ⚠️  实现结果有差异")
            except Exception as e:
                print(f"  结果验证失败: {e}")

def test_memory_usage():
    """测试内存使用情况"""
    print("\n=== 内存使用测试 ===")
    
    # 创建大图像测试内存使用
    test_img = torch.randint(0, 256, (3, 500, 500), dtype=torch.uint8).float()
    
    print(f"输入图像内存: {test_img.element_size() * test_img.numel() / 1024 / 1024:.2f} MB")
    
    # 测试优化实现的内存使用
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024
    
    result = media_conv(test_img)
    
    mem_after = process.memory_info().rss / 1024 / 1024
    mem_used = mem_after - mem_before
    
    print(f"优化实现内存使用: {mem_used:.2f} MB")
    print(f"输出图像内存: {result.element_size() * result.numel() / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    benchmark_median_filters()
    try:
        test_memory_usage()
    except ImportError:
        print("\n跳过内存测试（需要安装psutil包）") 