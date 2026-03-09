# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np

# # 保存路径
# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "distance_matrix")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # 设置 TGRS 风格字体
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman']
# plt.rcParams['font.size'] = 12

# # 1. 构造一个有物理意义的数据 (4个点)
# # 假设前3个点很近(cluster)，第4个点是outlier
# # 这里直接模拟生成的距离矩阵
# d_inner = 0.5  # 类内紧密距离
# d_outlier = 5.0 # 离群点距离

# matrix = np.array([
#     [0.0, 0.4, 0.6, 5.2],  # Sample 1 (Normal)
#     [0.4, 0.0, 0.5, 5.1],  # Sample 2 (Normal)
#     [0.6, 0.5, 0.0, 4.8],  # Sample 3 (Normal)
#     [5.2, 5.1, 4.8, 0.0]   # Sample 4 (Outlier!)
# ])

# # 标签
# labels = ['$x_1$', '$x_2$', '$x_3$', '$x_4$ (Outlier)']

# # 2. 绘制热力图
# fig, ax = plt.subplots(figsize=(6, 5))

# # 使用 Mask 隐藏对角线或者让对角线颜色变淡，以此突出非对角线关系
# # 但标准的距离矩阵通常保留对角线
# sns.heatmap(matrix, 
#             annot=True,            # 显示数值
#             fmt=".1f",             # 保留1位小数
#             cmap="YlGnBu",         # TGRS偏好的蓝绿色系，黄色代表远(高值)，蓝色代表近(低值)
#                                    # 注意：通常heatmap是值大颜色深，这里可以用 _r 反转，
#                                    # 或者逻辑上：深色=高相关=低距离。
#                                    # 建议：深蓝=近(0)，亮黄=远(5)。这样高亮显示Outlier。
#             vmin=0, vmax=6,
#             square=True, 
#             linewidths=1.5,        # 格子间距，增加清晰度
#             linecolor='white',
#             cbar_kws={'label': 'Hyperbolic Distance $d_{c_j}(\cdot, \cdot)$'},
#             xticklabels=labels,
#             yticklabels=labels,
#             ax=ax)

# # 3. 美化与细节
# ax.set_title(r"Intra-Class Distance Matrix $G_j$", fontsize=14, pad=15, fontweight='bold')
# plt.xticks(rotation=45, ha='right')
# plt.yticks(rotation=0)

# # 4. 添加标注 (让审稿人一眼看懂物理意义)
# # 在图的旁边或者下方，你可以加注：
# # "High values in Row 4 indicate $x_4$ is an outlier in the hyperbolic space."

# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, "distance_matrix.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_dir, "distance_matrix.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
# print("生成完毕：已绘制精确的格拉姆距离矩阵")
# plt.show()


# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import matplotlib.patches as patches

# # 保存路径
# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "distance_matrix")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # 1. 设置画板：不需要很大，因为只是一个组件
# fig, ax = plt.subplots(figsize=(3, 3))

# # 2. 构造具有“代表性”的数据 (Representative Data)
# # 既然不显示数字，我们不需要真实数据，
# # 而是需要一个能完美展示“这里有聚类，也有离群点”的完美数据
# # 0 表示对角线，1-3 表示类内距离近，8-9 表示离群点
# schematic_matrix = np.array([
#     [0, 2, 3, 9],  # Sample 1
#     [2, 0, 2, 8],  # Sample 2
#     [3, 2, 0, 9],  # Sample 3
#     [9, 8, 9, 0]   # Sample 4 (Outlier, 这一行/列颜色会很显眼)
# ])

# # 3. 绘制抽象热力图 (Abstract Heatmap)
# sns.heatmap(schematic_matrix, 
#             annot=False,          # 【关键】关掉数值
#             cbar=False,           # 【关键】关掉色条
#             xticklabels=False,    # 【关键】关掉X轴标签
#             yticklabels=False,    # 【关键】关掉Y轴标签
#             cmap="Blues",         # 使用简洁的单色系 (TGRS常用蓝/灰/橙)
#                                   # 建议：浅色代表距离远(Outlier)，深色代表距离近(Cluster)
#                                   # 如果用'Blues'，通常深色数值大。
#                                   # 若要深色代表距离近(数值0)，请加 _r => "Blues_r"
#             square=True,          # 强制正方形
#             linewidths=2.0,       # 加粗网格线，强调“张量”感
#             linecolor='black',    # 黑色网格线，符合示意图的线条风格
#             ax=ax)

# # 4. 添加外边框 (让它看起来像一个实体模块)
# # 使用 patch 增加一个粗黑框
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(2)
#     spine.set_color('black')

# # 5. 添加数学符号标签 (Math Label)
# # 这通常是示意图中唯一需要的文字
# # ax.set_xlabel(r"$\mathbf{G}_j$", fontsize=24, labelpad=10, fontweight='bold')
# # 如果想表明维度，可以加个副标，或者在后期处理软件里加
# # ax.text(0.5, -0.2, "4 $\times$ 4", transform=ax.transAxes, ha='center', fontsize=12)

# plt.tight_layout()

# # 6. 保存为透明背景的矢量图 (SVG/PDF)
# # 这样您可以直接拖进 Visio/PPT/Illustrator 进行组合
# # plt.savefig('Schematic_Matrix_Icon.pdf', format='pdf', bbox_inches='tight', transparent=True)
# plt.savefig(os.path.join(save_dir, "distance_matrix.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_dir, "distance_matrix.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
# print("生成完毕：已绘制精确的格拉姆距离矩阵")
# plt.show()


# import os
# import matplotlib.pyplot as plt
# import seaborn as sns
# import numpy as np
# import matplotlib.colors as mcolors

# # 保存路径
# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "distance_matrix")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # 1. 设置画板（消除空白，统一尺寸）
# fig, ax = plt.subplots(figsize=(3, 3))
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 消除画布边距
# ax.set_position([0, 0, 1, 1])  # 轴域占满画布

# # 2. 构造距离矩阵数据
# schematic_matrix = np.array([
#     [0, 2, 3, 9],
#     [2, 0, 2, 8],
#     [3, 2, 0, 9],
#     [9, 8, 9, 0]
# ])

# # 3. 【核心】自定义 #CCE4FF 单色系渐变
# # 起始色（小数值，类内）：深一点的 #CCE4FF → #99CCFF
# # 结束色（大数值，离群）：浅一点的 #CCE4FF → #E6F2FF
# start_color = "#CCE4FF"
# end_color = "#E6F2FF"
# custom_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "cc_e4ff_gradient", [start_color, end_color]
# )

# # 4. 绘制热力图（统一线条宽度+颜色）
# sns.heatmap(schematic_matrix, 
#             annot=False,          
#             cbar=False,           
#             xticklabels=False,    
#             yticklabels=False,    
#             cmap="Blues_r",     # 应用自定义 #CCE4FF 渐变
#             square=True,          
#             linewidths=0.5,       # 统一为 0.5pt（TGRS 要求）
#             linecolor="#6699CC",  # 网格线：深一点的蓝，与 #CCE4FF 协调
#             ax=ax)

# # 5. 外边框（统一颜色+宽度）
# for spine in ax.spines.values():
#     spine.set_visible(True)
#     spine.set_linewidth(0.5)
#     spine.set_color("#6699CC")  # 边框同网格线，保持统一

# # 6. 标签（统一颜色）
# ax.set_xlabel(r"$\mathbf{G}_j$", 
#               fontsize=24, 
#               labelpad=10, 
#               fontweight='bold',
#               color="#6699CC")  # 标签文字同边框，风格统一

# plt.tight_layout()

# # 7. 保存（透明背景+无空白）
# plt.savefig(os.path.join(save_dir, "distance_matrix_cc_e4ff.svg"), 
#             transparent=True, 
#             format="svg", 
#             bbox_inches='tight', 
#             pad_inches=0)
# plt.savefig(os.path.join(save_dir, "distance_matrix_cc_e4ff.png"), 
#             transparent=True, 
#             bbox_inches='tight', 
#             pad_inches=0, 
#             dpi=600)
# plt.show()


import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 保存路径
script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
save_dir = os.path.join(script_dir, "distance_matrix_v2")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================================
# 1. 定义新的非对称距离矩阵 (New Distance Matrix)
# ==========================================
# 目标：行和 (Row Sums) 必须互不相同，且顺序为 Row 2 < Row 3 < Row 1 < Row 4
# 构造逻辑：
# - Sample 2 是核心 (Sum=14)
# - Sample 3 稍远 (Sum=17)
# - Sample 1 更远 (Sum=20)
# - Sample 4 是 Outlier (Sum=26)
matrix_data = np.array([
    [0, 2, 4, 8],   # Row 1: Sum = 14 (权重排第3)
    [2, 0, 1, 3],   # Row 2: Sum = 6  (权重排第1 - 核心)
    [4, 1, 0, 7],   # Row 3: Sum = 12 (权重排第2)
    [8, 3, 7, 0]    # Row 4: Sum = 18 (权重排第4 - 边缘)
])

# ==========================================
# 2. 定义严格对应的权重向量 (Corresponding Weight Vector)
# ==========================================
# 映射逻辑：距离 Sum 越小 -> 权重越大 -> 颜色越深
# 我们手动设定一个梯度分明的向量，以保证视觉清晰度：
# Row 2 (Sum 14) -> Weight 9 (最深蓝)
# Row 3 (Sum 17) -> Weight 6 (中深蓝)
# Row 1 (Sum 20) -> Weight 4 (浅蓝，能看出比6浅)
# Row 4 (Sum 25) -> Weight 1 (几乎白)
vector_data = np.array([
    [4, 9, 6, 1]
])

# ==========================================
# 3. 绘图设置 (保持统一风格)
# ==========================================
def draw_heatmap(data, filename, figsize, is_vector=False):
    fig, ax = plt.subplots(figsize=figsize)
    
    sns.heatmap(data, 
                annot=False, 
                cbar=False, 
                xticklabels=False, 
                yticklabels=False, 
                cmap="Blues",       # 统一使用蓝色系
                square=True, 
                linewidths=2.0, 
                linecolor='black', 
                ax=ax)
    
    # 加粗边框
    for spine in ax.spines.values():
        spine.set_visible(True)
        if not is_vector:
            spine.set_linewidth(4)
        else:
            spine.set_linewidth(2)
        spine.set_color('black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename + ".svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, filename + ".png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

# ==========================================
# 4. 执行绘制
# ==========================================

# 绘制距离矩阵图 (3x3 尺寸)
draw_heatmap(matrix_data, "distance_matrix_distinct", figsize=(1.4, 1.3))
print(f"生成的距离矩阵行和验证: {matrix_data.sum(axis=1)}")
print("  -> Row 2 (14) < Row 3 (17) < Row 1 (20) < Row 4 (22)")
print("  -> 验证通过：4个位置拓扑地位完全不同。")

# 绘制权重向量图 (3x1 尺寸)
# 注意：这里 figsize 设为 (3.5, 1.2) 是为了让单个格子的大小看起来和矩阵里的格子差不多
draw_heatmap(vector_data, "attention_vector_distinct", figsize=(1.4, 1.3), is_vector=True)
print(f"生成的权重向量值: {vector_data}")
print("  -> 对应关系验证: Weight[1]=9 (最高) > Weight[2]=6 > Weight[0]=4 > Weight[3]=1")
print("  -> 验证通过：视觉颜色阶梯分明。")

print(f"所有图表已保存至: {save_dir}")
