# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Ellipse

# def draw_2d_feature_space():
#     # 保存路径
#     script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
#     save_dir = os.path.join(script_dir, "scatter")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 1. 设置画板
#     # figsize 设置为正方形或略宽，适应论文单栏或双栏布局
#     fig, ax = plt.subplots(figsize=(1.4, 1.3))
    
#     # 2. 数据配置
#     # 为了复刻原图布局，我们手动定义三个中心点
#     # 对应原图：左上(蓝圆)，右上(黄三角)，下方(绿五边形)
#     centers = {
#         'class_1': np.array([-2.5, 2.0]), 
#         'class_2': np.array([ 2.5, 2.0]),
#         'class_3': np.array([ 0.0, -2.5])
#     }
    
#     # 设置随机种子，保证每次生成的"随机"点位置固定，便于复现
#     np.random.seed(2026) 
    
#     # 3. 绘图参数配置 (为了与 TGRS 风格匹配)
#     scatter_kwargs = {
#         's': 150,           # 点的大小 (与3D图中的球体视觉大小尽量匹配)
#         'alpha': 0.9,       # 透明度
#         'edgecolors': 'k',  # 黑色描边，增加对比度
#         'linewidth': 0.5    # 描边线宽
#     }
    
#     ellipse_kwargs = {
#         'facecolor': 'none', # 只有边框
#         'edgecolor': '#333333', # 深灰色边框，比纯黑更优雅
#         'linestyle': '--',   # 虚线
#         'linewidth': 1.5,    # 虚线粗细
#         'alpha': 0.7         # 虚线透明度
#     }

#     # 4. 生成数据并绘制
    
#     # --- 类别 1: 蓝色圆形 (Blue Circles) ---
#     points_c1 = centers['class_1'] + np.random.normal(scale=0.5, size=(4, 2))
#     ax.scatter(points_c1[:, 0], points_c1[:, 1], 
#                c='#4169E1', marker='o', label='Class A', **scatter_kwargs)
    
#     # 添加虚线圈 (手动调整位置和大小以包裹数据)
#     ell_c1 = Ellipse(xy=centers['class_1'], width=4.5, height=3.0, angle=-15, **ellipse_kwargs)
#     ax.add_patch(ell_c1)


#     # --- 类别 2: 黄色三角形 (Yellow Triangles) ---
#     points_c2 = centers['class_2'] + np.random.normal(scale=0.5, size=(4, 2))
#     # ax.scatter(points_c2[:, 0], points_c2[:, 1], 
#     #            c='#FFD700', marker='^', label='Class B', **scatter_kwargs)
    
#     # 添加虚线圈
#     ell_c2 = Ellipse(xy=centers['class_2'], width=4.5, height=3.0, angle=15, **ellipse_kwargs)
#     # ax.add_patch(ell_c2)


#     # --- 类别 3: 绿色五边形 (Green Pentagons) ---
#     points_c3 = centers['class_3'] + np.random.normal(scale=0.5, size=(4, 2))
#     # ax.scatter(points_c3[:, 0], points_c3[:, 1], 
#     #            c='#3CB371', marker='p', label='Class C', **scatter_kwargs)
    
#     # 添加虚线圈
#     ell_c3 = Ellipse(xy=centers['class_3'], width=4.8, height=3.2, angle=0, **ellipse_kwargs)
#     # ax.add_patch(ell_c3)

#     # 5. 样式美化 (关键步骤：去除坐标轴，只保留几何关系)
#     ax.set_aspect('equal')
#     ax.axis('off') # 关闭坐标轴，因为这只是一个示意图
    
#     # 可选：如果希望有一个淡灰色的“平面”背景来暗示这是切平面，可以解开下面这行注释
#     # ax.add_patch(plt.Rectangle((-5, -5), 10, 10, color='#f0f0f0', zorder=0, alpha=0.3))

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "scatter_plot.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(save_dir, "scatter_plot.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# draw_2d_feature_space()


# import os
# import matplotlib.pyplot as plt
# import numpy as np
# from matplotlib.patches import Ellipse

# def generate_spread_points(center, n_points, radius_spread, min_dist):
#     """
#     生成互不重叠的随机点 (带最小距离约束)
#     :param center: 中心坐标 [x, y]
#     :param n_points: 点的数量
#     :param radius_spread: 分布的大致半径范围
#     :param min_dist: 点与点之间的最小距离 (防止重叠的关键)
#     :return: 坐标数组
#     """
#     points = []
#     attempts = 0
#     max_attempts = 1000 # 防止死循环
    
#     while len(points) < n_points and attempts < max_attempts:
#         # 1. 在中心周围生成候选点
#         # 使用均匀分布而不是正态分布，避免过度集中在中心
#         candidate = center + np.random.uniform(-radius_spread, radius_spread, size=2)
        
#         # 2. 检查距离
#         if len(points) == 0:
#             points.append(candidate)
#         else:
#             # 计算候选点到所有已存在点的距离
#             dists = np.linalg.norm(np.array(points) - candidate, axis=1)
#             # 只有当所有距离都大于 min_dist 时才接受
#             if np.all(dists >= min_dist):
#                 points.append(candidate)
        
#         attempts += 1
            
#     return np.array(points)

# def draw_2d_feature_space_v2():
#     # 保存路径
#     script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
#     save_dir = os.path.join(script_dir, "scatter")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # 1. 设置画板
#     fig, ax = plt.subplots(figsize=(6, 5))
    
#     # 2. 数据配置
#     # 稍微拉开一点中心距离，给点更多空间
#     centers = {
#         'class_1': np.array([-2.8, 2.2]), 
#         'class_2': np.array([ 2.8, 2.2]),
#         'class_3': np.array([ 0.0, -2.5])
#     }
    
#     # 设置随机种子 (调整这个数字直到你获得最完美的形状)
#     # 推荐尝试: 2026, 42, 88, 101
#     np.random.seed(2026) 
    
#     # 3. 关键参数微调
#     n_shots = 4         # 样本数
#     point_size = 180    # 点的大小
    
#     # === [核心控制参数] ===
#     spread_radius = 1.2 # 点分布的范围半径 (越大越散)
#     min_distance = 0.8  # 最小间距 (控制重叠：必须大于点直径的视觉比例)
    
#     scatter_kwargs = {
#         's': point_size,    
#         'alpha': 0.9,       
#         'edgecolors': 'k',  
#         'linewidth': 0.8    
#     }
    
#     ellipse_kwargs = {
#         'facecolor': 'none', 
#         'edgecolor': '#555555', 
#         'linestyle': '--',   
#         'linewidth': 1.5,    
#         'alpha': 0.6         
#     }

#     # 4. 生成数据并绘制 (使用新函数)
    
#     # --- 类别 1: 蓝色圆形 ---
#     points_c1 = generate_spread_points(centers['class_1'], n_shots, spread_radius, min_distance)
#     ax.scatter(points_c1[:, 0], points_c1[:, 1], 
#                c='#4169E1', marker='o', **scatter_kwargs)
#     # 虚线圈 (稍微调大一点以包住散开的点)
#     ax.add_patch(Ellipse(xy=centers['class_1'], width=4.8, height=3.5, angle=-20, **ellipse_kwargs))


#     # --- 类别 2: 黄色三角形 ---
#     points_c2 = generate_spread_points(centers['class_2'], n_shots, spread_radius, min_distance)
#     # ax.scatter(points_c2[:, 0], points_c2[:, 1], 
#     #            c='#FFD700', marker='^', **scatter_kwargs)
#     # ax.add_patch(Ellipse(xy=centers['class_2'], width=4.8, height=3.5, angle=20, **ellipse_kwargs))


#     # --- 类别 3: 绿色五边形 ---
#     points_c3 = generate_spread_points(centers['class_3'], n_shots, spread_radius, min_distance)
#     # ax.scatter(points_c3[:, 0], points_c3[:, 1], 
#     #            c='#3CB371', marker='p', **scatter_kwargs)
#     # ax.add_patch(Ellipse(xy=centers['class_3'], width=5.0, height=3.8, angle=0, **ellipse_kwargs))

#     # 5. 样式美化
#     ax.set_aspect('equal')
#     ax.axis('off') 
    
#     # 限制视野范围，保证留白
#     ax.set_xlim(-6, 6)
#     ax.set_ylim(-6, 6)

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "scatter_plot_v2.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(save_dir, "scatter_plot_v2.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# draw_2d_feature_space_v2()


import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

def generate_spread_points(center, n_points, radius_spread, min_dist):
    """
    (保持不变) 生成互不重叠的随机点
    """
    points = []
    attempts = 0
    max_attempts = 1000 
    
    while len(points) < n_points and attempts < max_attempts:
        candidate = center + np.random.uniform(-radius_spread, radius_spread, size=2)
        if len(points) == 0:
            points.append(candidate)
        else:
            dists = np.linalg.norm(np.array(points) - candidate, axis=1)
            if np.all(dists >= min_dist):
                points.append(candidate)
        attempts += 1
    return np.array(points)

def draw_adaptive_feature_space(target_class_indices=None):
    """
    自适应绘图函数
    :param target_class_indices: 一个列表，包含想绘制的类别索引 (例如 [0] 或 [0, 1])
                                 0: Blue, 1: Yellow, 2: Green
    """
    # 保存路径
    script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
    save_dir = os.path.join(script_dir, "scatter")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 默认绘制所有三个类
    if target_class_indices is None:
        target_class_indices = [0, 1, 2]
    
    n_classes = len(target_class_indices)
    
    # === 1. 自适应布局逻辑 (核心修改) ===
    # 根据要画的类别数量，决定它们摆在哪里
    if n_classes == 1:
        # 如果只有1类：直接放在正中心
        centers_layout = [np.array([0.0, 0.0])]
    elif n_classes == 2:
        # 如果有2类：左右对称分布
        centers_layout = [np.array([-3.0, 0.0]), np.array([3.0, 0.0])]
    else:
        # 如果有3类(或更多)：保持原来的三角分布
        centers_layout = [
            np.array([-2.8, 2.2]), # 左上
            np.array([ 2.8, 2.2]), # 右上
            np.array([ 0.0, -2.5]) # 下方
        ]
        
    # 定义每个类别的样式库 (颜色, 形状, 标签)
    styles = [
        {'c': '#CCE4FF', 'm': 'o', 'label': 'Class A'}, # 0: 蓝圆
        {'c': '#FFD700', 'm': '^', 'label': 'Class B'}, # 1: 黄三角
        {'c': '#3CB371', 'm': 'p', 'label': 'Class C'}  # 2: 绿五边形
    ]

    # === 2. 绘图配置 ===
    fig, ax = plt.subplots(figsize=(6, 5), dpi=300) # 保持宽高比
    np.random.seed(2022) 
    
    n_shots = 4         
    point_size = 90    
    spread_radius = 1.0 # 稍微减小一点分布半径，适配圆形
    min_distance = 0.8  
    circle_radius = 1.7 # 圆的半径
    
    # 记录边界以便后续自动缩放
    all_x = []
    all_y = []

    # === 3. 循环绘制 ===
    # zip将“分配好的位置”和“被选中的类别的样式”对应起来
    for layout_pos, class_idx in zip(centers_layout, target_class_indices):
        style = styles[class_idx]
        
        # A. 生成点
        points = generate_spread_points(layout_pos, n_shots, spread_radius, min_distance)
        
        # B. 绘制散点
        ax.scatter(points[:, 0], points[:, 1], 
                   c=style['c'], marker=style['m'], 
                   s=point_size, alpha=1.0, edgecolors='black', linewidth=0.5)
        
        # C. 绘制虚线圆 (替代原来的 Ellipse)
        # 使用 matplotlib.patches.Circle
        circle = Circle(xy=layout_pos, radius=circle_radius,
                        facecolor='none', edgecolor='#000000',
                        linestyle='--', linewidth=2.0, alpha=1.0)
        ax.add_patch(circle)
        
        # D. 收集边界信息
        all_x.append(layout_pos[0] - circle_radius)
        all_x.append(layout_pos[0] + circle_radius)
        all_y.append(layout_pos[1] - circle_radius)
        all_y.append(layout_pos[1] + circle_radius)

    # === 4. 自适应视口 (填充画布) ===
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 计算所有圆的最边缘坐标
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    
    # 添加一点边距 (padding)，以免圆切到画布边缘
    padding = 0.5
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "scatter_plot_v3.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, "scatter_plot_v3.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.show()

# ==========================================
# 在这里修改你想画的类别！
# 试图生成只有 "Class A (蓝色)" 的情况:
# ==========================================
print("生成单个类别 (居中，填充画布):")
draw_adaptive_feature_space(target_class_indices=[0])

# 如果你想画两个类，请取消下面这行的注释:
draw_adaptive_feature_space(target_class_indices=[0, 1])

# 如果你想画原本的三个类，请取消下面这行的注释:
draw_adaptive_feature_space(target_class_indices=[0, 1, 2])