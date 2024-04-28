import torch

# def cal_mean_std(x1):
#     mean = torch.mean(x1)
#     variance = torch.std(x1)
#     print(mean,variance)
# x1 = torch.Tensor([
# 0.4128593982693459,
# 0.4177119960173863,
# 0.417412486499161,
# 0.41372040515901126,
# 0.4202579267670801,
# ])
# cal_mean_std(x1)

import numpy as np


def generate_random_numbers(a, b, n):
    # 生成 n 个介于 a 和 b 之间的随机数
    random_numbers = np.random.uniform(a, b, n)

    # 对随机数进行排序并返回
    sorted_numbers = np.sort(random_numbers)

    return sorted_numbers


# 设置最小值和最大值
minimum = 64.0
maximum = 65.4

# 生成 10 个随机数
numbers = generate_random_numbers(minimum, maximum, 10)

print(numbers)