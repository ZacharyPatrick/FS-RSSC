import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import ListedColormap

# ==========================================
# 0. 基础设置
# ==========================================
script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
save_dir = os.path.join(script_dir, "attention_vector")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================================
# 1. 用户自定义颜色设置 (User Defined Colors)
# ==========================================
# 在这里指定4个格子的颜色（从左到右）
# 您可以使用 Hex 颜色码 (如 "#FFFFFF") 或颜色名称 (如 "red")
# 下面的颜色对应原本的权重逻辑：
# 格子1 (Row1, 权重4): 浅蓝
# 格子2 (Row2, 权重9): 深蓝 (核心)
# 格子3 (Row3, 权重6): 中蓝
# 格子4 (Row4, 权重1): 极浅/白 (边缘)

sj = [
    "#80C8C8",  # 格子 1 的颜色（灰蓝）
    "#FA7F6F",  # 格子 2 的颜色（三文鱼红）
    "#80C8C8",  # 格子 3 的颜色（灰蓝）
    "#FDD865"   # 格子 4 的颜色（明黄）
]

wj = [
    "#A8D0D0",  # 格子 1 的颜色
    "#F9C7B8",  # 格子 2 的颜色
    "#C8E6E6",  # 格子 3 的颜色
    "#F9E8B8"   # 格子 4 的颜色
]

# 单格子颜色配置（可自由修改，示例为灰蓝+三文鱼红混合色）
single_grid_color = [
    "#BDA49C"  # 单个格子的颜色（灰蓝+三文鱼红混合色，也可替换为你需要的任意色号）
]

# 构造绘图数据：
# 4格向量：索引 [0, 1, 2, 3] 对应4个颜色
vector_indices_4grid = np.array([[0, 1, 2, 3]])
# 单格向量：索引 [0] 对应单个颜色
vector_indices_1grid = np.array([[0]])

# ==========================================
# 2. 绘图函数
# ==========================================
def draw_custom_color_vector(data_indices, color_list, filename, figsize, is_single=False):
    fig, ax = plt.subplots(figsize=figsize)
    
    # 创建自定义的 Colormap
    custom_cmap = ListedColormap(color_list)
    
    sns.heatmap(data_indices, 
                annot=False, 
                cbar=False, 
                xticklabels=False, 
                yticklabels=False, 
                cmap=custom_cmap,   # 使用自定义颜色列表
                square=True, 
                linewidths=2.0, 
                linecolor='black', 
                ax=ax)
    
    # 设置边框样式 (对应原代码中 is_vector=True 的逻辑，线宽为2)
    for spine in ax.spines.values():
        spine.set_visible(True)
        if not is_single:
            spine.set_linewidth(2)  # 保持原代码中向量图的边框宽度
        else:
            spine.set_linewidth(4)
        spine.set_color('black')
    
    plt.tight_layout()
    
    # 保存图片
    svg_path = os.path.join(save_dir, filename + ".svg")
    png_path = os.path.join(save_dir, filename + ".png")
    
    plt.savefig(svg_path, transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(png_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()
    print(f"图表已保存至: {svg_path}")

# ==========================================
# 3. 执行绘制
# ==========================================

# 保持原代码中的尺寸设置 figsize=(1.4, 1.3)
# draw_custom_color_vector(vector_indices, sj, "attention_vector_sj", figsize=(1.4, 1.3))

# draw_custom_color_vector(vector_indices_4grid, wj, "attention_vector_wj", figsize=(1.4, 1.3))

# 绘制单格子向量（核心新增功能）
# figsize=(0.35, 1.3)：宽度为4格的1/4，高度一致，保证单个格子大小和4格中的一致
draw_custom_color_vector(vector_indices_1grid, single_grid_color, "attention_vector_single_grid", figsize=(1.4, 1.3), is_single=True)

print("=== 绘制完成 ===")
# print(f"当前使用的颜色顺序: {sj}")
print(f"当前使用的颜色顺序: {wj}")
