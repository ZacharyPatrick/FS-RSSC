import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib.patches as patches

# 保存路径 (保持与您一致)
script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
save_dir = os.path.join(script_dir, "distance_matrix")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 1. 设置画板
# 向量通常是扁长的。既然矩阵是 4x4 (正方形)，向量是 1x4 (长条)。
# 为了保证格子大小看起来和矩阵里的格子差不多大，我们需要调整 figsize。
# 矩阵是 (3, 3)，这里设为 (3, 1) 左右比较合适。
fig, ax = plt.subplots(figsize=(3.5, 1.2))

# 2. 构造与距离矩阵严格对应的权重数据 (Representative Weights)
# 逻辑推导：
# 距离矩阵 Row 1 (Cluster): Sum=14 -> 权重 High (设为 7)
# 距离矩阵 Row 2 (Medoid):  Sum=12 -> 权重 Highest (设为 9，最深色)
# 距离矩阵 Row 3 (Cluster): Sum=14 -> 权重 High (设为 7)
# 距离矩阵 Row 4 (Outlier): Sum=26 -> 权重 Very Low (设为 1，最浅色)
# 注意：这里的数据形状必须是 (1, 4) 的二维数组，才能被 heatmap 识别为单行向量
schematic_vector = np.array([
    [7, 9, 5, 1] 
])

# 3. 绘制抽象热力图 (Abstract Heatmap)
sns.heatmap(schematic_vector, 
            annot=False,          # 关掉数值
            cbar=False,           # 关掉色条
            xticklabels=False,    # 关掉X轴标签
            yticklabels=False,    # 关掉Y轴标签
            cmap="Blues",         # 【关键】保持配色统一
                                  # 逻辑：数值大(9) -> 颜色深 -> 权重高 -> 代表重要样本
                                  # 逻辑：数值小(1) -> 颜色浅 -> 权重低 -> 代表被抑制的离群点
                                  # 这与距离矩阵的视觉逻辑（深色=大值）在“数值”层面是一致的，
                                  # 但在“物理含义”上成功反转了（矩阵深色是坏的，向量深色是好的）。
            square=True,          # 强制每个单元格为正方形
            linewidths=2.0,       # 保持一致的边框粗细
            linecolor='black',    # 保持一致的边框颜色
            ax=ax)

# 4. 添加外边框
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(2)
    spine.set_color('black')

# 5. 添加数学符号标签 (可选)
# ax.set_xlabel(r"$\mathbf{a}_j$", fontsize=24, labelpad=10, fontweight='bold')

plt.tight_layout()

# 6. 保存
plt.savefig(os.path.join(save_dir, "attention_vector_aj.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(save_dir, "attention_vector_aj.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)

print("生成完毕：已绘制与距离矩阵逻辑严格对应的 Attention 权重向量")
plt.show()