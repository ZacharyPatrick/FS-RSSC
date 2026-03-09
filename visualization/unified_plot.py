import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, proj3d
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Circle

# ==========================================
# [全局配置] 核心视觉参数 (以3D图为基准)
# ==========================================
# 画布尺寸 (保持两图一致，确保物理尺寸相同)
FIG_SIZE = (1.4, 1.3)
DPI = 300

# 节点统一样式
UNIFIED_NODE_SIZE = 9       # 统一节点大小 (原脚本为9)
UNIFIED_LINE_WIDTH = 0.5    # 统一边缘线宽
UNIFIED_EDGE_COLOR = 'black'
NODE_COLOR_DEFAULT = "#CCE4FF" # 默认淡蓝色 (Class A)

# 颜色定义
COLOR_SURF_START = "#151515"
COLOR_SURF_END = "#FAFAFA"
ARROW_COLOR = "#000000"

# 路径设置
script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
save_dir = os.path.join(script_dir, "unified_imgs")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ==========================================
# [模块 1] 3D 绘图工具类与函数
# ==========================================

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        target_color = kwargs.pop('color', 'black')
        if 'facecolor' not in kwargs: kwargs['facecolor'] = target_color
        if 'edgecolor' not in kwargs: kwargs['edgecolor'] = target_color
        kwargs['fill'] = True
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['lw'] = 0.8
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

def draw_lorentz_surface():
    """绘制洛伦兹双曲面图 (3D)"""
    print("正在绘制: 洛伦兹双曲面 (3D)...")
    
    # --- 参数设置 ---
    START_OFFSET = 0.25
    END_OFFSET = 0.35
    line_width_surf = 0.05
    
    # --- 120°方位角计算 ---
    azim_angle = 120  
    theta = np.radians(azim_angle)
    cos_theta, sin_theta = np.cos(theta), np.sin(theta)

    visual_quadrants = [
        (-1.3, 1.3),   # 视觉左上
        (1.3, 1.3),    # 视觉右上
        (-1.5, -12.5/6), # 视觉左下
        (2.0, -0.5/6)  # 视觉右下
    ]
    
    original_coords = []
    for x_vis, y_vis in visual_quadrants:
        x_ori = x_vis * cos_theta + y_vis * sin_theta
        y_ori = -x_vis * sin_theta + y_vis * cos_theta
        original_coords.append((x_ori, y_ori))

    # --- 生成曲面数据 ---
    x = np.linspace(-3, 3, 120)
    y = np.linspace(-3, 3, 120)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(1 + X**2 + Y**2)

    # --- 初始化画布 ---
    fig = plt.figure(figsize=FIG_SIZE, dpi=DPI)
    fig.patch.set_alpha(0.0)
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)
    ax.set_axis_off()

    # --- 绘制曲面 ---
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("surf_gradient", [COLOR_SURF_START, COLOR_SURF_END])
    ax.plot_surface(X, Y, Z, cmap=custom_cmap, edgecolor=(0, 0, 0, 0.15),
                    rstride=4, cstride=4, alpha=0.4, shade=False,
                    linewidth=line_width_surf, antialiased=True, zorder=1)

    # --- 生成节点 ---
    center_node = np.array([0, 0, np.sqrt(1)])
    sub_nodes = []
    for x_ori, y_ori in original_coords:
        surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.35
        sub_nodes.append(np.array([x_ori, y_ori, surface_z]))

    # --- 绘制箭头 ---
    for sub_node in sub_nodes:
        start_point = sub_node
        end_point = center_node
        vec = end_point - start_point
        dist = np.linalg.norm(vec)
        if dist < START_OFFSET + END_OFFSET: continue
        unit_vec = vec / dist 
        new_start = start_point + unit_vec * START_OFFSET
        new_end = end_point - unit_vec * min(END_OFFSET, dist * 0.4)
        
        arrow = Arrow3D([new_start[0], new_end[0]], [new_start[1], new_end[1]], [new_start[2], new_end[2]], 
                        mutation_scale=2, lw=0.5, arrowstyle="-|>", color=ARROW_COLOR, zorder=20)
        # ax.add_artist(arrow) # 如果需要箭头，取消注释

    # --- 绘制节点 (使用全局统一大小) ---
    # 中心点
    # ax.scatter3D(center_node[0], center_node[1], center_node[2], 
    #              c=NODE_COLOR_DEFAULT, s=UNIFIED_NODE_SIZE, 
    #              edgecolors=UNIFIED_EDGE_COLOR, linewidth=UNIFIED_LINE_WIDTH, marker='^', zorder=30)
    
    # 周围节点
    for node in sub_nodes:
        ax.scatter3D(node[0], node[1], node[2], 
                     c=NODE_COLOR_DEFAULT, s=UNIFIED_NODE_SIZE, 
                     edgecolors=UNIFIED_EDGE_COLOR, linewidth=UNIFIED_LINE_WIDTH, zorder=30)

    # --- 视角调整 ---
    ax.view_init(elev=25, azim=120)
    ax.set_zlim(1, 3.5)
    ax.set_box_aspect((1, 1, 0.6))
    ax.dist = 5.8
    
    # --- 保存 ---
    plt.savefig(os.path.join(save_dir, "plot_1_hyperbolic_3d.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, "plot_1_hyperbolic_3d.png"), bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)
    plt.close() # 释放内存

# ==========================================
# [模块 2] 2D 散点图工具函数
# ==========================================

def generate_spread_points(center, n_points, radius_spread, min_dist):
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

def draw_scatter_2d(target_class_indices=None):
    """绘制自适应散点图 (2D)"""
    print("正在绘制: 特征散点图 (2D)...")
    
    if target_class_indices is None:
        target_class_indices = [0] # 默认画一个

    # 布局逻辑
    n_classes = len(target_class_indices)
    if n_classes == 1:
        centers_layout = [np.array([0.0, 0.0])]
    elif n_classes == 2:
        centers_layout = [np.array([-3.0, 0.0]), np.array([3.0, 0.0])]
    else:
        centers_layout = [np.array([-2.8, 2.2]), np.array([2.8, 2.2]), np.array([0.0, -2.5])]

    # 样式库 (注意：Class A 的颜色与 3D 图一致)
    styles = [
        {'c': NODE_COLOR_DEFAULT, 'm': 'o', 'label': 'Class A'}, # 0: 与3D图一致的淡蓝
        {'c': '#FFD700', 'm': '^', 'label': 'Class B'}, # 1: 黄
        {'c': '#3CB371', 'm': 'p', 'label': 'Class C'}  # 2: 绿
    ]

    # --- 初始化画布 (使用相同的 FIG_SIZE) ---
    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)
    np.random.seed(2026)
    
    n_shots = 5
    spread_radius = 1.0
    min_distance = 0.8
    circle_radius = 1.7
    
    all_x, all_y = [], []

    # --- 绘图循环 ---
    for layout_pos, class_idx in zip(centers_layout, target_class_indices):
        style = styles[class_idx]
        points = generate_spread_points(layout_pos, n_shots, spread_radius, min_distance)
        
        # [关键] 这里使用全局定义的 UNIFIED_NODE_SIZE
        ax.scatter(points[:, 0], points[:, 1], 
                   c=style['c'], marker=style['m'], 
                   s=UNIFIED_NODE_SIZE,  # <--- 强制统一大小
                   alpha=1.0, 
                   edgecolors=UNIFIED_EDGE_COLOR, 
                   linewidth=UNIFIED_LINE_WIDTH) # <--- 强制统一线宽
        
        circle = Circle(xy=layout_pos, radius=circle_radius,
                        facecolor='none', edgecolor='#000000',
                        linestyle='--', linewidth=2.0, alpha=1.0)
        ax.add_patch(circle)
        
        all_x.extend([layout_pos[0] - circle_radius, layout_pos[0] + circle_radius])
        all_y.extend([layout_pos[1] - circle_radius, layout_pos[1] + circle_radius])

    # --- 自适应调整 ---
    ax.set_aspect('equal')
    ax.axis('off')
    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    padding = 0.5
    ax.set_xlim(min_x - padding, max_x + padding)
    ax.set_ylim(min_y - padding, max_y + padding)

    plt.tight_layout()
    # 根据类别数量给文件命名
    suffix = "_".join(map(str, target_class_indices))
    plt.savefig(os.path.join(save_dir, f"plot_2_scatter_{suffix}.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, f"plot_2_scatter_{suffix}.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()

# ==========================================
# [主执行逻辑]
# ==========================================
if __name__ == "__main__":
    print("=== 开始生成统一视觉图表 ===")
    
    # 1. 生成 3D 洛伦兹图 (基准)
    draw_lorentz_surface()
    
    # 2. 生成 2D 散点图 (大小将自动匹配)
    #    你可以选择画几个类，例如 [0] 或 [0, 1]
    draw_scatter_2d(target_class_indices=[0])      # 单类
    draw_scatter_2d(target_class_indices=[0, 1])   # 双类
    
    print(f"=== 完成！图片已保存至: {save_dir} ===")
    