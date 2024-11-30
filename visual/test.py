import matplotlib.pyplot as plt
import numpy as np
import sys

import os


# 假设 size 和 part 数组
size = 40  # 示例大小
part = data = np.loadtxt('partition.txt')  # 示例的 part 数组，随机生成 0, 1, 或 2

# 重塑 part 数组为 2D 网格
grid = part.reshape((size, size))

# 创建颜色映射
# 使用 colormap 来为不同的组指定颜色
unique_groups = np.unique(grid)
colors = plt.cm.get_cmap('rainbow', len(unique_groups))

# 绘制网格
plt.imshow(grid, cmap=colors, origin='upper')

# 添加颜色条
cbar = plt.colorbar()
cbar.set_label('Group')

# 设置标题和标签
plt.title('Grid Visualization of Groups')

# 显示图形
plt.show()