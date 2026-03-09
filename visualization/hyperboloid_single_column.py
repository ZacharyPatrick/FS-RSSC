# import os
# from matplotlib import markers
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d

# # ================= 0. TGRS 配色方案 =================
# COLOR_MANIFOLD_BOTTOM = "#151515"   # 曲面底部颜色 (白色渐变)
# COLOR_MANIFOLD_TOP    = "#FAFAFA"   # 曲面顶部颜色 (鼠尾草绿) - 几何容器
# COLOR_POINT_SUPPORT   = "#CCE4FF"   # 正样本 (天青蓝)
# COLOR_POINT_EDGE      = "#606060"   # 点的边缘色 (深灰，比纯黑柔和)
# COLOR_PROTO_FILL      = "#FDD865"   # 原型填充 (明黄)
# COLOR_PROTO_EDGE      = "#C00000"   # 原型边缘 (深红) - 强调核心
# COLOR_LINK            = "#000000"   # 连线颜色 (深灰)

# # --- 【核心修改】自定义的 3D 箭头类 ---
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         # 1. 获取用户想要的主色调 (例如 'black')，如果没有则默认为黑色
#         target_color = kwargs.pop('color', 'black')
        
#         # 2. 【关键步骤】强制设置 facecolor 和 edgecolor 相同
#         # 这样能保证箭头是实心的，不会出现空心描边的情况
#         if 'facecolor' not in kwargs:
#             kwargs['facecolor'] = target_color
#         if 'edgecolor' not in kwargs:
#             kwargs['edgecolor'] = target_color
            
#         # 3. 确保填充被激活
#         kwargs['fill'] = True
        
#         # 4. 稍微调整线宽，让箭头边缘更锐利
#         if 'lw' not in kwargs and 'linewidth' not in kwargs:
#              kwargs['lw'] = 0  # 如果边缘和填充同色，设为0可以避免边缘锯齿

#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def do_3d_projection(self, renderer=None):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
#         # 计算投影后的 2D 坐标，更新箭头位置
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         return np.min(zs)

# # ================= 1. 工具函数：生成碗内点 =================
# def generate_cluster(center_x, center_y, num_points=4, spread=0.15, random_seed=None):
#     if random_seed:
#         np.random.seed(random_seed)
#     x = np.random.normal(center_x, spread * 4.0, num_points)
#     y = np.random.normal(center_y, spread * 2.0, num_points)
#     # 保持较大的Z轴偏移，确保点悬浮在碗内
#     z = np.sqrt(1 + x**2 + y**2) + 0.8 # 稍微降低一点高度，让它贴近曲面
#     return x, y, z

# # ================= 2. 数学工具：双曲几何计算 =================
# def minkowski_dot(u, v):
#     return u[0]*v[0] + u[1]*v[1] - u[2]*v[2]

# def get_geodesic_path(start_p, end_p, num_points=30, stop_ratio=0.0):
#     prod = minkowski_dot(start_p, end_p)
#     if prod > -1.0: 
#         prod = -1.0 - 1e-7
#     d = np.arccosh(-prod)
    
#     if d < 1e-6:
#         return np.tile(start_p, (num_points, 1)).T

#     t_end = 1.0 - stop_ratio
#     t = np.linspace(0, t_end, num_points)
    
#     sinh_d = np.sinh(d)
#     coeff_start = np.sinh((1 - t) * d) / sinh_d
#     coeff_end = np.sinh(t * d) / sinh_d
    
#     path = np.outer(coeff_start, start_p) + np.outer(coeff_end, end_p)
#     return path[:, 0], path[:, 1], path[:, 2]

# # ================= 3. 绘制双曲测地线箭头 =================
# def draw_geodesic_arrows(ax, start_xs, start_ys, start_zs, end_x, end_y, end_z, color='black'):
#     # # 针对小图调整停止距离和箭头长度
#     # stop_distance_buffer = 0.5  # 增大一点，防止箭头戳进星星里
#     # arrow_len = 0.25            # 相对长度增加，因为坐标轴看起来更紧凑
    
#     # end_vec = np.array([end_x, end_y, end_z])

#     # for sx, sy, sz in zip(start_xs, start_ys, start_zs):
#     #     start_vec = np.array([sx, sy, sz])
        
#     #     prod = minkowski_dot(start_vec, end_vec)
#     #     if prod > -1.0: prod = -1.0 - 1e-7
#     #     total_dist = np.arccosh(-prod)
        
#     #     if total_dist < stop_distance_buffer: continue

#     #     current_stop_ratio = stop_distance_buffer / total_dist
#     #     gx, gy, gz = get_geodesic_path(start_vec, end_vec, num_points=60, stop_ratio=current_stop_ratio)
        
#     #     # 绘制虚线 (加粗)
#     #     ax.plot(gx, gy, gz, 
#     #             color=color, 
#     #             linestyle=(0, (3, 1.5)), # 稍微密一点的虚线
#     #             linewidth=1.0,           # TGRS: 加粗线条
#     #             alpha=0.9, 
#     #             zorder=300)
        
#     #     # 绘制箭头
#     #     if len(gx) >= 2:
#     #         dx = gx[-1] - gx[-2]
#     #         dy = gy[-1] - gy[-2]
#     #         dz = gz[-1] - gz[-2]
#     #         norm = np.sqrt(dx**2 + dy**2 + dz**2)
#     #         if norm > 0:
#     #             ux, uy, uz = dx/norm, dy/norm, dz/norm
#     #         else:
#     #             ux, uy, uz = 0, 0, 0

#     #         ax.quiver(gx[-1], gy[-1], gz[-1], ux, uy, uz, 
#     #                   length=arrow_len, normalize=True,
#     #                   color=color, 
#     #                   arrow_length_ratio=0.3, # 箭头头大一点
#     #                   linewidth=1.0,          # TGRS: 加粗箭头
#     #                   alpha=1.0, 
#     #                   zorder=301)
#     # 调整停止距离，稍微留一点点空隙给星星，但不要太大
#     stop_distance_buffer = 0.7 
    
#     end_vec = np.array([end_x, end_y, end_z])

#     for sx, sy, sz in zip(start_xs, start_ys, start_zs):
#         start_vec = np.array([sx, sy, sz])
        
#         prod = minkowski_dot(start_vec, end_vec)
#         if prod > -1.0: prod = -1.0 - 1e-7
#         total_dist = np.arccosh(-prod)
        
#         if total_dist < stop_distance_buffer: continue

#         # 计算路径
#         current_stop_ratio = stop_distance_buffer / total_dist
#         gx, gy, gz = get_geodesic_path(start_vec, end_vec, num_points=60, stop_ratio=current_stop_ratio)
        
#         # --- 1. 画虚线 (路径主体) ---
#         # 注意：这里我们画到倒数第2个点，最后一段留给箭头头
#         ax.plot(gx[:], gy[:], gz[:], 
#                 color=color, 
#                 linestyle=(0, (3, 1.2)), # 稍微紧凑一点的虚线
#                 linewidth=0.8,           # 线宽要精致
#                 alpha=0.9, 
#                 zorder=300)
        
#         # --- 2. 画箭头头 (使用自定义 Arrow3D) ---
#         # 取路径最后两个点，定义箭头的方向
#         arrow_start_x, arrow_start_y, arrow_start_z = gx[-2], gy[-2], gz[-2]
#         arrow_end_x,   arrow_end_y,   arrow_end_z   = gx[-1], gy[-1], gz[-1]

#         # 创建一个扁平的 3D 箭头
#         # arrowstyle='-|>' 是标准的 PPT 三角箭头样式
#         # mutation_scale 控制箭头大小
#         arrow = Arrow3D([arrow_start_x, arrow_end_x], 
#                         [arrow_start_y, arrow_end_y], 
#                         [arrow_start_z, arrow_end_z], 
#                         mutation_scale=2,  # 【微调】稍微加大了一点箭头，让它更明显
#                         arrowstyle="-|>",   # 实心三角头样式
#                         color=color,        # 这里的颜色现在会被正确应用到填充和边框
#                         alpha=1.0,          # 不透明
#                         zorder=301)
#         ax.add_artist(arrow)

# # ================= 4. 主绘图函数 (TGRS 单栏版) =================
# def plot_hyperbolic_single_column():
#     script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
#     save_dir = os.path.join(script_dir, "tgrs_single_column")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # --- 关键参数调整 (适应 3.5 英寸宽度) ---
#     figsize = (1.4, 1.3)       # TGRS 单栏标准尺寸
#     view_elev = 25             # 稍微提高视角，看清内部
#     view_azim = 120            # 旋转角度
#     xy_limit = 3.0             # 缩小范围，让物体看起来更大
    
#     # 视觉元素大小
#     grid_linewidth = 0.05       # 网格线加粗
#     surface_alpha = 0.3        # 透明度
#     point_size = 20           # 样本点大小 (看起来要大)
#     proto_size = 100           # 原型五角星大小 (必须醒目)
    
#     # 自定义 Colormap 
#     cmap_colors = [COLOR_MANIFOLD_BOTTOM, COLOR_MANIFOLD_TOP]
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list("hyperbolic_fade", cmap_colors)

#     # --- 初始化画布 ---
#     fig = plt.figure(figsize=figsize, dpi=300) # 高 DPI 预览
#     # 调整边距，尽可能利用 3.5 英寸的所有空间
#     plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
#     # ax = fig.add_subplot(111, projection='3d')
#     # --- 【关键修改 1】使用 add_axes 占满 100% 画布 ---
#     # [left, bottom, width, height] 全部设为 0 到 1
#     # 这比 subplots_adjust 更彻底，它不给坐标轴留任何物理空间
#     ax = fig.add_axes([0, 0, 1, 1], projection='3d')
#     ax.set_axis_off()          # 关闭坐标轴
    
#     # --- 绘制洛伦兹曲面 (碗) ---
#     x = np.linspace(-xy_limit, xy_limit, 120) # 降低分辨率以减少网格密集度
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     ax.plot_surface(X, Y, Z, 
#                     cmap=custom_cmap,
#                     edgecolor=(0, 0, 0, 0.15), # 网格颜色
#                     alpha=surface_alpha,
#                     rstride=4, cstride=4,     # 网格稀疏一点，避免小图糊成一团
#                     linewidth=grid_linewidth,
#                     antialiased=True,
#                     shade=False,
#                     zorder=1)

#     # --- 生成数据 (Class A + Proto) ---
#     # --- 类别 A（黄色，中心）---
#     cx_a, cy_a = 0.0, 0.0
#     feat_x_a, feat_y_a, feat_z_a = generate_cluster(cx_a, cy_a, num_points=4, spread=0.2, random_seed=2)
#     # 原型位于正下方
#     proto_z_a = np.sqrt(1 + cx_a**2 + cy_a**2) 

#     # --- 类别 B（红色，右侧）---
#     cx_b_1, cy_b_1 = -0.2, 0.7
#     feat_x_b_1, feat_y_b_1, feat_z_b_1 = generate_cluster(cx_b_1, cy_b_1, num_points=1, spread=0.15, random_seed=1)
#     cx_b_2, cy_b_2 = -1.9, -1.5
#     feat_x_b_2, feat_y_b_2, feat_z_b_2 = generate_cluster(cx_b_2, cy_b_2, num_points=1, spread=0.15, random_seed=1)
    
#     # --- 类别 C（蓝色，左侧）---
#     cx_c, cy_c = -1.0, 0.5
#     feat_x_c, feat_y_c, feat_z_c = generate_cluster(cx_c, cy_c, num_points=1, spread=0.15, random_seed=1)

#     # --- 类别 D（绿色，左侧）---
#     cx_d, cy_d = 0.5, 2.0
#     feat_x_d, feat_y_d, feat_z_d = generate_cluster(cx_d, cy_d, num_points=1, spread=0.10, random_seed=1)

#     # --- 绘制连线 (放在点之前绘制，不遮挡点) ---
#     draw_geodesic_arrows(ax, feat_x_a, feat_y_a, feat_z_a, cx_a, cy_a, proto_z_a, color=COLOR_LINK)

#     # --- 绘制样本点 (天青蓝) ---
#     ax.scatter3D(feat_x_a, feat_y_a, feat_z_a, s=point_size, c="#CCE4FF", marker='o', 
#                  edgecolors="#606060", linewidth=0.6, zorder=100, label='Class A')
#     # ax.scatter3D(feat_x_b_1, feat_y_b_1, feat_z_b_1, s=point_size, c='#E53935', marker='o', 
#     #            edgecolors='#000000', linewidth=0.8, zorder=100, label='Class B')
#     # ax.scatter3D(feat_x_b_2, feat_y_b_2, feat_z_b_2, s=point_size, c='#E53935', marker='o', 
#     #            edgecolors='#000000', linewidth=0.8, zorder=100, label='Class B')
#     # ax.scatter3D(feat_x_c, feat_y_c, feat_z_c, s=point_size, c='#1E88E5', marker='o', 
#     #            edgecolors='#000000', linewidth=0.8, zorder=100, label='Class C')
#     # ax.scatter3D(feat_x_d, feat_y_d, feat_z_d, s=point_size, c='#43A047', marker='o', 
#     #            edgecolors='#000000', linewidth=0.8, zorder=100, label='Class D')

#     # --- 绘制原型 (明黄 + 深红描边) ---
#     ax.scatter3D([cx_a], [cy_a], [proto_z_a], s=proto_size, c="#CCE4FF", marker='^', 
#                  edgecolors="#606060", linewidth=1.0, zorder=200)

#     # --- 相机与比例设置 ---
#     ax.view_init(elev=view_elev, azim=view_azim)
    
#     # 关键：调整 ax.dist 来放大物体 (Zoom In)
#     # Matplotlib 默认 dist 是 ~10。设置小一点可以放大物体填满画布
#     ax.dist = 5.8 
    
#     ax.set_zlim(1, 3.5)
#     # 拉伸 Z 轴，让碗看起来更深，符合 Lorentz 模型的视觉特征
#     ax.set_box_aspect((1, 1, 0.6)) 

#     # --- 保存 ---
#     # SVG 格式用于插入 PPT/Visio (矢量，无损放大)
#     plt.savefig(os.path.join(save_dir, "HCPR_module_single_col_1.svg"), format="svg", transparent=True, bbox_inches='tight', pad_inches=0)
#     # PNG 用于快速预览
#     plt.savefig(os.path.join(save_dir, "HCPR_module_single_col_1.png"), dpi=600, transparent=True, bbox_inches='tight', pad_inches=0)
    
#     print(f"生成完毕：单栏图已保存至 {save_dir}")
#     print("请在 PPT 中插入 SVG 文件，并将宽度设置为 8.9cm (3.5 inch)。")
#     plt.show()

# if __name__ == "__main__":
#     plot_hyperbolic_single_column()


# 洛伦兹曲面1
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d

# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "imgs")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # ================= 1. 自定义3D箭头类 =================
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         target_color = kwargs.pop('color', 'black')
#         if 'facecolor' not in kwargs:
#             kwargs['facecolor'] = target_color
#         if 'edgecolor' not in kwargs:
#             kwargs['edgecolor'] = target_color
#         kwargs['fill'] = True  # 箭头实心
#         if 'lw' not in kwargs and 'linewidth' not in kwargs:
#             kwargs['lw'] = 0.8
#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def do_3d_projection(self, renderer=None):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         return np.min(zs)

# # ================= 2. 核心参数与旋转计算 =================
# COLOR_SURF_START = "#151515"  
# COLOR_SURF_END = "#FAFAFA"    
# NODE_COLORS = ["#C71585", "#87CEEB", "#FFA500", "#98FB98", "#9370DB"]
# ARROW_COLOR = "#000000"

# START_OFFSET = 0.25
# END_OFFSET = 0.35
# STOP_RATIO = 0.1  
# NODE_OFFSET = 0.12
# xy_limit = 3.0
# line_width = 0.00

# # --- 120°方位角下的视觉象限坐标计算 ---
# azim_angle = 120  
# theta = np.radians(azim_angle)
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)

# visual_quadrants = [
#     (-0.7, 1.4),   # 视觉左上
#     (1.3, 1.3),    # 视觉右上
#     (-1.5, -12.5/6),  # 视觉左下
#     (2.0, -0.5/6)    # 视觉右下
# ]

# original_coords = []
# for x_vis, y_vis in visual_quadrants:
#     x_ori = x_vis * cos_theta + y_vis * sin_theta
#     y_ori = -x_vis * sin_theta + y_vis * cos_theta
#     original_coords.append((x_ori, y_ori))

# # ================= 3. 生成洛伦兹双曲面上叶 =================
# x = np.linspace(-3, 3, 120)
# y = np.linspace(-3, 3, 120)
# X, Y = np.meshgrid(x, y)
# scaling_factor = 1.0  
# Z = np.sqrt(1 + (scaling_factor * X)**2 + (scaling_factor * Y)**2)

# # ================= 4. 初始化画布与绘图 =================
# fig = plt.figure(figsize=(1.4, 1.3), dpi=300)
# fig.patch.set_alpha(0.0)
# ax = fig.add_subplot(111, projection='3d')
# ax.patch.set_alpha(0.0)
# ax.zaxis.set_rotate_label(False)
# ax.xaxis.set_rotate_label(False)
# ax.yaxis.set_rotate_label(False)
# ax.set_axis_off()

# custom_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "surf_gradient", [COLOR_SURF_START, COLOR_SURF_END]
# )
# ax.plot_surface(
#     X, Y, Z,
#     cmap=custom_cmap,
#     edgecolor=(0, 0, 0, 0.15),
#     rstride=4, cstride=4,
#     alpha=0.4,
#     shade=False,
#     linewidth=line_width,
#     antialiased=True,
#     zorder=1
# )

# # ================= 5. 生成节点坐标 =================
# center_node = np.array([0, 0, np.sqrt(1 + 0**2 + 0**2)])
# sub_nodes = []
# for i, (x_ori, y_ori) in enumerate(original_coords):
#     surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.35
#     z_ori = surface_z * 1.0
#     sub_nodes.append(np.array([x_ori, y_ori, z_ori]))

# # 【核心新增】3个手绘红色圆圈的节点（对应图片位置）
# # 红圈1：左上节点右侧
# # red1_x_vis, red1_y_vis = (-0.08, 2.0)
# # red1_x_ori = red1_x_vis * cos_theta + red1_y_vis * sin_theta
# # red1_y_ori = -red1_x_vis * sin_theta + red1_y_vis * cos_theta
# # red1_z_ori = np.sqrt(1 + red1_x_ori**2 + red1_y_ori**2) + 0.35
# # sub_nodes.append(np.array([red1_x_ori, red1_y_ori, red1_z_ori]))

# # 红圈2：中心节点左下方
# # red2_x_vis, red2_y_vis = (-1.9, -0.5*1.5)
# # red2_x_ori = red2_x_vis * cos_theta + red2_y_vis * sin_theta
# # red2_y_ori = -red2_x_vis * sin_theta + red2_y_vis * cos_theta
# # red2_z_ori = np.sqrt(1 + red2_x_ori**2 + red2_y_ori**2) + 0.07
# # sub_nodes.append(np.array([red2_x_ori, red2_y_ori, red2_z_ori]))

# # 红圈3：右下节点左下方
# # red3_x_vis, red3_y_vis = (1.2*0.6, -0.6)
# # red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
# # red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
# # red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.27
# # sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# # 红圈4：右上节点右上方
# # red3_x_vis, red3_y_vis = (1.7, 0.7)
# # red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
# # red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
# # red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.35
# # sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# all_nodes = [center_node] + sub_nodes

# TARGET_INDICES = []  # 需加长箭头的子节点索引（根据你的需求改）

# # ================= 6. 绘制节点与3D箭头 =================
# # 绘制所有子节点的箭头（包含新红色节点）
# for i, sub_node in enumerate(sub_nodes):
#     start_point = sub_node
#     end_point = center_node
    
#     vec = end_point - start_point
#     dist = np.linalg.norm(vec)
#     if dist < START_OFFSET + END_OFFSET:
#         continue
#     unit_vec = vec / dist      

#     # 【新增条件判断：目标节点用更小的偏移，箭头更长】
#     if i in TARGET_INDICES:
#         current_start_offset = 0.10  # 更小的起点偏移（原0.25）
#         current_end_offset = 0.00   # 更小的终点偏移（原0.35）
#     else:
#         current_start_offset = START_OFFSET  # 原有节点用默认值
#         current_end_offset = END_OFFSET
    
#     new_start = start_point + unit_vec * current_start_offset
#     actual_end_offset = min(current_end_offset, dist * 0.4) 
#     new_end = end_point - unit_vec * actual_end_offset
    
#     arrow = Arrow3D(
#         [new_start[0], new_end[0]], 
#         [new_start[1], new_end[1]], 
#         [new_start[2], new_end[2]], 
#         mutation_scale=2,
#         lw=0.5, 
#         arrowstyle="-|>", 
#         color=ARROW_COLOR, 
#         zorder=20
#     )
#     # ax.add_artist(arrow)

# # 绘制中心点
# # ax.scatter3D(center_node[0], center_node[1], center_node[2], 
# #              color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, marker='^', zorder=30)

# # 绘制子节点（区分原有节点和新红色节点）
# for i, node in enumerate(sub_nodes):
#     ax.scatter3D(node[0], node[1], node[2], 
#                     color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, zorder=30)

# # ================= 7. 调整视角与样式 =================
# ax.view_init(elev=25, azim=120)
# ax.set_zlim(1, 3.5)
# ax.set_box_aspect((1, 1, 0.6))
# ax.dist = 5.8
# ax.draw(renderer=fig.canvas.get_renderer())

# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow.png"), bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)
# plt.show()


# 洛伦兹曲面2
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d

# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "imgs")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # ================= 1. 自定义3D箭头类 =================
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         target_color = kwargs.pop('color', 'black')
#         if 'facecolor' not in kwargs:
#             kwargs['facecolor'] = target_color
#         if 'edgecolor' not in kwargs:
#             kwargs['edgecolor'] = target_color
#         kwargs['fill'] = True  # 箭头实心
#         if 'lw' not in kwargs and 'linewidth' not in kwargs:
#             kwargs['lw'] = 0.8
#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def do_3d_projection(self, renderer=None):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         return np.min(zs)

# # ================= 2. 核心参数与旋转计算 =================
# COLOR_SURF_START = "#151515"  
# COLOR_SURF_END = "#FAFAFA"    
# NODE_COLORS = ["#C71585", "#87CEEB", "#FFA500", "#98FB98", "#9370DB"]
# ARROW_COLOR = "#000000"

# START_OFFSET = 0.25
# END_OFFSET = 0.35
# STOP_RATIO = 0.1  
# NODE_OFFSET = 0.12
# xy_limit = 3.0
# line_width = 0.00

# # --- 120°方位角下的视觉象限坐标计算 ---
# azim_angle = 120  
# theta = np.radians(azim_angle)
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)

# visual_quadrants = [
#     (-0.7, 1.4),   # 视觉左上
#     (1.3, 1.3),    # 视觉右上
#     (-1.5, -12.5/6),  # 视觉左下
#     (2.0, -0.5/6)    # 视觉右下
# ]

# original_coords = []
# for x_vis, y_vis in visual_quadrants:
#     x_ori = x_vis * cos_theta + y_vis * sin_theta
#     y_ori = -x_vis * sin_theta + y_vis * cos_theta
#     original_coords.append((x_ori, y_ori))

# # ================= 3. 生成洛伦兹双曲面上叶 =================
# x = np.linspace(-3, 3, 120)
# y = np.linspace(-3, 3, 120)
# X, Y = np.meshgrid(x, y)
# scaling_factor = 1.0  
# Z = np.sqrt(1 + (scaling_factor * X)**2 + (scaling_factor * Y)**2)

# # ================= 4. 初始化画布与绘图 =================
# fig = plt.figure(figsize=(1.4, 1.3), dpi=300)
# fig.patch.set_alpha(0.0)
# ax = fig.add_subplot(111, projection='3d')
# ax.patch.set_alpha(0.0)
# ax.zaxis.set_rotate_label(False)
# ax.xaxis.set_rotate_label(False)
# ax.yaxis.set_rotate_label(False)
# ax.set_axis_off()

# custom_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "surf_gradient", [COLOR_SURF_START, COLOR_SURF_END]
# )
# ax.plot_surface(
#     X, Y, Z,
#     cmap=custom_cmap,
#     edgecolor=(0, 0, 0, 0.15),
#     rstride=4, cstride=4,
#     alpha=0.4,
#     shade=False,
#     linewidth=line_width,
#     antialiased=True,
#     zorder=1
# )

# # ================= 5. 生成节点坐标 =================
# center_node = np.array([0, 0, np.sqrt(1 + 0**2 + 0**2)])
# sub_nodes = []
# for i, (x_ori, y_ori) in enumerate(original_coords):
#     surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.35
#     z_ori = surface_z * 1.0
#     sub_nodes.append(np.array([x_ori, y_ori, z_ori]))

# # 【核心新增】3个手绘红色圆圈的节点（对应图片位置）
# # 红圈1：左上节点右侧
# # red1_x_vis, red1_y_vis = (-0.08, 2.0)
# # red1_x_ori = red1_x_vis * cos_theta + red1_y_vis * sin_theta
# # red1_y_ori = -red1_x_vis * sin_theta + red1_y_vis * cos_theta
# # red1_z_ori = np.sqrt(1 + red1_x_ori**2 + red1_y_ori**2) + 0.35
# # sub_nodes.append(np.array([red1_x_ori, red1_y_ori, red1_z_ori]))

# # 红圈2：中心节点左下方
# # red2_x_vis, red2_y_vis = (-1.9, -0.5*1.5)
# # red2_x_ori = red2_x_vis * cos_theta + red2_y_vis * sin_theta
# # red2_y_ori = -red2_x_vis * sin_theta + red2_y_vis * cos_theta
# # red2_z_ori = np.sqrt(1 + red2_x_ori**2 + red2_y_ori**2) + 0.07
# # sub_nodes.append(np.array([red2_x_ori, red2_y_ori, red2_z_ori]))

# # 红圈3：右下节点左下方
# # red3_x_vis, red3_y_vis = (1.2*0.6, -0.6)
# # red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
# # red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
# # red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.27
# # sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# # 红圈4：右上节点右上方
# # red3_x_vis, red3_y_vis = (1.7, 0.7)
# # red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
# # red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
# # red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.35
# # sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# all_nodes = [center_node] + sub_nodes

# TARGET_INDICES = []  # 需加长箭头的子节点索引（根据你的需求改）

# # ================= 6. 绘制节点与3D箭头 =================
# # 绘制所有子节点的箭头（包含新红色节点）
# for i, sub_node in enumerate(sub_nodes):
#     start_point = sub_node
#     end_point = center_node
    
#     vec = end_point - start_point
#     dist = np.linalg.norm(vec)
#     if dist < START_OFFSET + END_OFFSET:
#         continue
#     unit_vec = vec / dist      

#     # 【新增条件判断：目标节点用更小的偏移，箭头更长】
#     if i in TARGET_INDICES:
#         current_start_offset = 0.10  # 更小的起点偏移（原0.25）
#         current_end_offset = 0.00   # 更小的终点偏移（原0.35）
#     else:
#         current_start_offset = START_OFFSET  # 原有节点用默认值
#         current_end_offset = END_OFFSET
    
#     new_start = start_point + unit_vec * current_start_offset
#     actual_end_offset = min(current_end_offset, dist * 0.4) 
#     new_end = end_point - unit_vec * actual_end_offset
    
#     arrow = Arrow3D(
#         [new_start[0], new_end[0]], 
#         [new_start[1], new_end[1]], 
#         [new_start[2], new_end[2]], 
#         mutation_scale=2,
#         lw=0.5, 
#         arrowstyle="-|>", 
#         color=ARROW_COLOR, 
#         zorder=20
#     )
#     ax.add_artist(arrow)

# # 绘制中心点
# ax.scatter3D(center_node[0], center_node[1], center_node[2], 
#              color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, marker='^', zorder=30)

# # 绘制子节点（区分原有节点和新红色节点）
# for i, node in enumerate(sub_nodes):
#     ax.scatter3D(node[0], node[1], node[2], 
#                     color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, zorder=30)

# # ================= 7. 调整视角与样式 =================
# ax.view_init(elev=25, azim=120)
# ax.set_zlim(1, 3.5)
# ax.set_box_aspect((1, 1, 0.6))
# ax.dist = 5.8
# ax.draw(renderer=fig.canvas.get_renderer())

# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow2.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow2.png"), bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)
# plt.show()


# 洛伦兹曲面3
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
save_dir = os.path.join(script_dir, "imgs")
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ================= 1. 自定义3D箭头类 =================
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        target_color = kwargs.pop('color', 'black')
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = target_color
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = target_color
        kwargs['fill'] = True  # 箭头实心
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['lw'] = 0.8
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# ================= 2. 核心参数与旋转计算 =================
COLOR_SURF_START = "#151515"  
COLOR_SURF_END = "#FAFAFA"    
NODE_COLORS = ["#C71585", "#87CEEB", "#FFA500", "#98FB98", "#9370DB"]
ARROW_COLOR = "#000000"

START_OFFSET = 0.25
END_OFFSET = 0.35
STOP_RATIO = 0.1  
NODE_OFFSET = 0.12
xy_limit = 3.0
line_width = 0.00

# --- 120°方位角下的视觉象限坐标计算 ---
azim_angle = 120  
theta = np.radians(azim_angle)
cos_theta = np.cos(theta)
sin_theta = np.sin(theta)

visual_quadrants = [
    (-0.7, 1.4),   # 视觉左上
    (1.3, 1.3),    # 视觉右上
    (-1.5, -12.5/6),  # 视觉左下
    (2.0, -0.5/6)    # 视觉右下
]

original_coords = []
for x_vis, y_vis in visual_quadrants:
    x_ori = x_vis * cos_theta + y_vis * sin_theta
    y_ori = -x_vis * sin_theta + y_vis * cos_theta
    original_coords.append((x_ori, y_ori))

# ================= 3. 生成洛伦兹双曲面上叶 =================
x = np.linspace(-3, 3, 120)
y = np.linspace(-3, 3, 120)
X, Y = np.meshgrid(x, y)
scaling_factor = 1.0  
Z = np.sqrt(1 + (scaling_factor * X)**2 + (scaling_factor * Y)**2)

# ================= 4. 初始化画布与绘图 =================
fig = plt.figure(figsize=(1.4, 1.3), dpi=300)
fig.patch.set_alpha(0.0)
ax = fig.add_subplot(111, projection='3d')
ax.patch.set_alpha(0.0)
ax.zaxis.set_rotate_label(False)
ax.xaxis.set_rotate_label(False)
ax.yaxis.set_rotate_label(False)
ax.set_axis_off()

custom_cmap = mcolors.LinearSegmentedColormap.from_list(
    "surf_gradient", [COLOR_SURF_START, COLOR_SURF_END]
)
ax.plot_surface(
    X, Y, Z,
    cmap=custom_cmap,
    edgecolor=(0, 0, 0, 0.15),
    rstride=4, cstride=4,
    alpha=0.4,
    shade=False,
    linewidth=line_width,
    antialiased=True,
    zorder=1
)

# ================= 5. 生成节点坐标 =================
center_node = np.array([0, 0, np.sqrt(1 + 0**2 + 0**2)])
sub_nodes = []
for i, (x_ori, y_ori) in enumerate(original_coords):
    surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.35
    z_ori = surface_z * 1.0
    sub_nodes.append(np.array([x_ori, y_ori, z_ori]))

# 【核心新增】3个手绘红色圆圈的节点（对应图片位置）
# 红圈1：左上节点右侧（三文鱼红）
red1_x_vis, red1_y_vis = (-0.08, 2.0)
red1_x_ori = red1_x_vis * cos_theta + red1_y_vis * sin_theta
red1_y_ori = -red1_x_vis * sin_theta + red1_y_vis * cos_theta
red1_z_ori = np.sqrt(1 + red1_x_ori**2 + red1_y_ori**2) + 0.35
sub_nodes.append(np.array([red1_x_ori, red1_y_ori, red1_z_ori]))

# 红圈2：中心节点左下方（灰蓝）
red2_x_vis, red2_y_vis = (-1.9, -0.5*1.5)
red2_x_ori = red2_x_vis * cos_theta + red2_y_vis * sin_theta
red2_y_ori = -red2_x_vis * sin_theta + red2_y_vis * cos_theta
red2_z_ori = np.sqrt(1 + red2_x_ori**2 + red2_y_ori**2) + 0.07
sub_nodes.append(np.array([red2_x_ori, red2_y_ori, red2_z_ori]))

# 红圈3：右下节点左下方（灰蓝）
red3_x_vis, red3_y_vis = (1.2*0.6, -0.6)
red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.27
sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# 红圈4：右上节点右上方（明黄）
red3_x_vis, red3_y_vis = (1.7, 0.7)
red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.35
sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# 【核心新增】8个同类别子节点（3个#FA7F6F，3个#FDD865，2个#80C8C8）
# 视觉坐标分布在不同区域，避免遮挡
new_visual_quadrants = [
    # 3个#FA7F6F（三文鱼红）
    (-1.9, 1.9),   # 左上外围
    (-1.8, -3.6),  # 左下外围
    (2.7, -0.1),   # 中下区域
    # 3个#FDD865（明黄）
    (1.4, 2.5),    # 右上外围
    (2.2, 1.1),    # 右中区域
    (0.8, 1.8),    # 右上内层
    # 2个#80C8C8（灰蓝）
    (0.5, 3.0),   # 左中区域
    (0.9, 2.3),   # 右下外围
]

for x_vis, y_vis in new_visual_quadrants:
    x_ori = x_vis * cos_theta + y_vis * sin_theta
    y_ori = -x_vis * sin_theta + y_vis * cos_theta
    surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.2  # 统一z偏移，避免嵌入
    z_ori = surface_z * 1.0
    sub_nodes.append(np.array([x_ori, y_ori, z_ori]))

all_nodes = [center_node] + sub_nodes

TARGET_INDICES = [6]  # 需加长箭头的子节点索引（根据你的需求改）

# ================= 6. 绘制节点与3D箭头 =================
# 绘制所有子节点的箭头（包含新红色节点）
for i, sub_node in enumerate(sub_nodes):
    start_point = sub_node
    end_point = center_node
    
    vec = end_point - start_point
    dist = np.linalg.norm(vec)
    if dist < START_OFFSET + END_OFFSET:
        continue
    unit_vec = vec / dist      

    # 【新增条件判断：目标节点用更小的偏移，箭头更长】
    if i in TARGET_INDICES:
        current_start_offset = 0.10  # 更小的起点偏移（原0.25）
        current_end_offset = 0.00   # 更小的终点偏移（原0.35）
    else:
        current_start_offset = START_OFFSET  # 原有节点用默认值
        current_end_offset = END_OFFSET
    
    new_start = start_point + unit_vec * current_start_offset
    actual_end_offset = min(current_end_offset, dist * 0.4) 
    new_end = end_point - unit_vec * actual_end_offset
    
    arrow = Arrow3D(
        [new_start[0], new_end[0]], 
        [new_start[1], new_end[1]], 
        [new_start[2], new_end[2]], 
        mutation_scale=2,
        lw=0.5, 
        arrowstyle="-|>", 
        color=ARROW_COLOR, 
        zorder=20
    )
    # ax.add_artist(arrow)

# 绘制中心点
ax.scatter3D(center_node[0], center_node[1], center_node[2], 
             color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, marker='^', zorder=30)

# 绘制子节点（区分原有节点和新红色节点）
for i, node in enumerate(sub_nodes):
    if 5 <= i <= 6:
        ax.scatter3D(node[0], node[1], node[2], 
                     color="#80C8C8", s=9, edgecolors='black', linewidth=1.0, zorder=30)
    elif i == 4:
        ax.scatter3D(node[0], node[1], node[2], 
                     color="#FA7F6F", s=9, edgecolors='black', linewidth=1.0, zorder=30)
    elif i == 7:
        ax.scatter3D(node[0], node[1], node[2], 
                     color="#FDD865", s=9, edgecolors='black', linewidth=1.0, zorder=30)
    # 新增8个分类节点（索引8-15）
    elif 8 <= i <= 10:  # 3个#FA7F6F
        ax.scatter3D(node[0], node[1], node[2], 
                     color="#FA7F6F", s=9, edgecolors='black', linewidth=0.5, zorder=30)
    elif 11 <= i <= 13:  # 3个#FDD865
        ax.scatter3D(node[0], node[1], node[2], 
                     color="#FDD865", s=9, edgecolors='black', linewidth=0.5, zorder=30)
    elif 14 <= i <= 15:  # 2个#80C8C8
        ax.scatter3D(node[0], node[1], node[2], 
                     color="#80C8C8", s=9, edgecolors='black', linewidth=0.5, zorder=30)
    else:
        ax.scatter3D(node[0], node[1], node[2], 
                     color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, zorder=30)

# ================= 7. 调整视角与样式 =================
ax.view_init(elev=25, azim=120)
ax.set_zlim(1, 3.5)
ax.set_box_aspect((1, 1, 0.6))
ax.dist = 5.8
ax.draw(renderer=fig.canvas.get_renderer())

plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow3.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow3.png"), bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)
plt.show()


# 洛伦兹曲面4
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d

# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "imgs")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # ================= 1. 自定义3D箭头类 =================
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         target_color = kwargs.pop('color', 'black')
#         if 'facecolor' not in kwargs:
#             kwargs['facecolor'] = target_color
#         if 'edgecolor' not in kwargs:
#             kwargs['edgecolor'] = target_color
#         kwargs['fill'] = True  # 箭头实心
#         if 'lw' not in kwargs and 'linewidth' not in kwargs:
#             kwargs['lw'] = 0.8
#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def do_3d_projection(self, renderer=None):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         return np.min(zs)

# # ================= 2. 核心参数与旋转计算 =================
# COLOR_SURF_START = "#151515"  
# COLOR_SURF_END = "#FAFAFA"    
# NODE_COLORS = ["#C71585", "#87CEEB", "#FFA500", "#98FB98", "#9370DB"]
# ARROW_COLOR = "#000000"

# START_OFFSET = 0.25
# END_OFFSET = 0.35
# STOP_RATIO = 0.1  
# NODE_OFFSET = 0.12
# xy_limit = 3.0
# line_width = 0.00

# # --- 120°方位角下的视觉象限坐标计算 ---
# azim_angle = 120  
# theta = np.radians(azim_angle)
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)

# visual_quadrants = [
#     (-0.7, 1.4),   # 视觉左上
#     (1.3, 1.3),    # 视觉右上
#     (-1.5, -12.5/6),  # 视觉左下
#     (2.0, -0.5/6)    # 视觉右下
# ]

# original_coords = []
# for x_vis, y_vis in visual_quadrants:
#     x_ori = x_vis * cos_theta + y_vis * sin_theta
#     y_ori = -x_vis * sin_theta + y_vis * cos_theta
#     original_coords.append((x_ori, y_ori))

# # ================= 3. 生成洛伦兹双曲面上叶 =================
# x = np.linspace(-3, 3, 120)
# y = np.linspace(-3, 3, 120)
# X, Y = np.meshgrid(x, y)
# scaling_factor = 1.0  
# Z = np.sqrt(1 + (scaling_factor * X)**2 + (scaling_factor * Y)**2)

# # ================= 4. 初始化画布与绘图 =================
# fig = plt.figure(figsize=(1.4, 1.3), dpi=300)
# fig.patch.set_alpha(0.0)
# ax = fig.add_subplot(111, projection='3d')
# ax.patch.set_alpha(0.0)
# ax.zaxis.set_rotate_label(False)
# ax.xaxis.set_rotate_label(False)
# ax.yaxis.set_rotate_label(False)
# ax.set_axis_off()

# custom_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "surf_gradient", [COLOR_SURF_START, COLOR_SURF_END]
# )
# ax.plot_surface(
#     X, Y, Z,
#     cmap=custom_cmap,
#     edgecolor=(0, 0, 0, 0.15),
#     rstride=4, cstride=4,
#     alpha=0.4,
#     shade=False,
#     linewidth=line_width,
#     antialiased=True,
#     zorder=1
# )

# # ================= 5. 生成节点坐标 =================
# center_node = np.array([0, 0, np.sqrt(1 + 0**2 + 0**2)])
# sub_nodes = []
# for i, (x_ori, y_ori) in enumerate(original_coords):
#     surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.35
#     z_ori = surface_z * 1.0
#     sub_nodes.append(np.array([x_ori, y_ori, z_ori]))

# # 【核心新增】3个手绘红色圆圈的节点（对应图片位置）
# # 红圈1：左上节点右侧
# red1_x_vis, red1_y_vis = (-0.08, 2.0)
# red1_x_ori = red1_x_vis * cos_theta + red1_y_vis * sin_theta
# red1_y_ori = -red1_x_vis * sin_theta + red1_y_vis * cos_theta
# red1_z_ori = np.sqrt(1 + red1_x_ori**2 + red1_y_ori**2) + 0.35
# sub_nodes.append(np.array([red1_x_ori, red1_y_ori, red1_z_ori]))

# # 红圈2：中心节点左下方
# red2_x_vis, red2_y_vis = (-1.9, -0.5*1.5)
# red2_x_ori = red2_x_vis * cos_theta + red2_y_vis * sin_theta
# red2_y_ori = -red2_x_vis * sin_theta + red2_y_vis * cos_theta
# red2_z_ori = np.sqrt(1 + red2_x_ori**2 + red2_y_ori**2) + 0.07
# sub_nodes.append(np.array([red2_x_ori, red2_y_ori, red2_z_ori]))

# # 红圈3：右下节点左下方
# red3_x_vis, red3_y_vis = (1.2*0.6, -0.6)
# red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
# red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
# red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.27
# sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# # 红圈4：右上节点右上方
# red3_x_vis, red3_y_vis = (1.7, 0.7)
# red3_x_ori = red3_x_vis * cos_theta + red3_y_vis * sin_theta
# red3_y_ori = -red3_x_vis * sin_theta + red3_y_vis * cos_theta
# red3_z_ori = np.sqrt(1 + red3_x_ori**2 + red3_y_ori**2) + 0.35
# sub_nodes.append(np.array([red3_x_ori, red3_y_ori, red3_z_ori]))

# all_nodes = [center_node] + sub_nodes

# TARGET_INDICES = [6]  # 需加长箭头的子节点索引（根据你的需求改）

# # ================= 6. 绘制节点与3D箭头 =================
# # 绘制所有子节点的箭头（包含新红色节点）
# for i, sub_node in enumerate(sub_nodes):
#     start_point = sub_node
#     end_point = center_node
    
#     vec = end_point - start_point
#     dist = np.linalg.norm(vec)
#     if dist < START_OFFSET + END_OFFSET:
#         continue
#     unit_vec = vec / dist      

#     # 【新增条件判断：目标节点用更小的偏移，箭头更长】
#     if i in TARGET_INDICES:
#         current_start_offset = 0.10  # 更小的起点偏移（原0.25）
#         current_end_offset = 0.00   # 更小的终点偏移（原0.35）
#     else:
#         current_start_offset = START_OFFSET  # 原有节点用默认值
#         current_end_offset = END_OFFSET
    
#     new_start = start_point + unit_vec * current_start_offset
#     actual_end_offset = min(current_end_offset, dist * 0.4) 
#     new_end = end_point - unit_vec * actual_end_offset
    
#     arrow = Arrow3D(
#         [new_start[0], new_end[0]], 
#         [new_start[1], new_end[1]], 
#         [new_start[2], new_end[2]], 
#         mutation_scale=2,
#         lw=0.5, 
#         arrowstyle="-|>", 
#         color=ARROW_COLOR, 
#         zorder=20
#     )
#     ax.add_artist(arrow)

# # 绘制中心点
# ax.scatter3D(center_node[0], center_node[1], center_node[2], 
#              color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, marker='d', zorder=30)

# # 绘制子节点（区分原有节点和新红色节点）
# for i, node in enumerate(sub_nodes):
#     if i == 5 or i == 6:
#         ax.scatter3D(node[0], node[1], node[2], 
#                      color="#80C8C8", s=9, edgecolors='black', linewidth=1.0, zorder=30)
#     elif i == 4:
#         ax.scatter3D(node[0], node[1], node[2], 
#                      color="#FA7F6F", s=9, edgecolors='black', linewidth=1.0, zorder=30)
#     elif i == 7:
#         ax.scatter3D(node[0], node[1], node[2], 
#                      color="#FDD865", s=9, edgecolors='black', linewidth=1.0, zorder=30)
#     else:
#         ax.scatter3D(node[0], node[1], node[2], 
#                      color="#CCE4FF", s=9, edgecolors='black', linewidth=0.5, zorder=30)

# # ================= 7. 调整视角与样式 =================
# ax.view_init(elev=25, azim=120)
# ax.set_zlim(1, 3.5)
# ax.set_box_aspect((1, 1, 0.6))
# ax.dist = 5.8
# ax.draw(renderer=fig.canvas.get_renderer())

# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow4.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow4.png"), bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)
# plt.show()


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d

# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "imgs")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # ================= 1. 自定义3D箭头类 =================
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         target_color = kwargs.pop('color', 'black')
#         if 'facecolor' not in kwargs:
#             kwargs['facecolor'] = target_color
#         if 'edgecolor' not in kwargs:
#             kwargs['edgecolor'] = target_color
#         kwargs['fill'] = True  # 箭头实心
#         if 'lw' not in kwargs and 'linewidth' not in kwargs:
#             kwargs['lw'] = 0.8
#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def do_3d_projection(self, renderer=None):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         return np.min(zs)

# # ================= 2. 核心参数与旋转计算 =================
# COLOR_SURF_START = "#151515"  
# COLOR_SURF_END = "#FAFAFA"    
# NODE_COLORS = ["#C71585", "#87CEEB", "#FFA500", "#98FB98", "#9370DB"]
# ARROW_COLOR = "#000000"

# START_OFFSET = 0.25
# END_OFFSET = 0.35
# STOP_RATIO = 0.1  
# NODE_OFFSET = 0.12
# xy_limit = 3.0
# line_width = 0.05

# # --- 120°方位角下的视觉象限坐标计算 ---
# azim_angle = 120  
# theta = np.radians(azim_angle)
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)

# visual_quadrants = [
#     (-1.3, 1.3),   # 视觉左上
#     (1.3, 1.3),    # 视觉右上
#     (-1.5, -12.5/6),  # 视觉左下
#     (1.5, -2.5/6)    # 视觉右下
# ]

# original_coords = []
# for x_vis, y_vis in visual_quadrants:
#     x_ori = x_vis * cos_theta + y_vis * sin_theta
#     y_ori = -x_vis * sin_theta + y_vis * cos_theta
#     original_coords.append((x_ori, y_ori))

# # ================= 3. 生成洛伦兹双曲面上叶 =================
# x = np.linspace(-3, 3, 120)
# y = np.linspace(-3, 3, 120)
# X, Y = np.meshgrid(x, y)
# scaling_factor = 1.0  
# Z = np.sqrt(1 + (scaling_factor * X)**2 + (scaling_factor * Y)**2)

# # ================= 4. 初始化画布与绘图 =================
# fig = plt.figure(figsize=(1.4, 1.3), dpi=300)
# fig.patch.set_alpha(0.0)
# ax = fig.add_subplot(111, projection='3d')
# ax.patch.set_alpha(0.0)
# ax.zaxis.set_rotate_label(False)
# ax.xaxis.set_rotate_label(False)
# ax.yaxis.set_rotate_label(False)
# ax.set_axis_off()

# custom_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "surf_gradient", [COLOR_SURF_START, COLOR_SURF_END]
# )
# ax.plot_surface(
#     X, Y, Z,
#     cmap=custom_cmap,
#     edgecolor=(0, 0, 0, 0.15),
#     rstride=4, cstride=4,
#     alpha=0.4,
#     shade=False,
#     linewidth=line_width,
#     antialiased=True,
#     zorder=1
# )

# # ================= 5. 生成节点坐标 =================
# center_node = np.array([0, 0, np.sqrt(1 + 0**2 + 0**2)])
# sub_nodes = []
# for i, (x_ori, y_ori) in enumerate(original_coords):
#     surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.2
#     z_ori = surface_z * 1.0
#     sub_nodes.append(np.array([x_ori, y_ori, z_ori]))

# # 【核心新增】红色圆圈位置的节点（右上区域，原右上节点右上方）
# new_x_vis, new_y_vis = 2.0, 0.5  # 视觉坐标，精准匹配红色圆圈位置
# new_x_ori = new_x_vis * cos_theta + new_y_vis * sin_theta
# new_y_ori = -new_x_vis * sin_theta + new_y_vis * cos_theta
# new_surface_z = np.sqrt(1 + new_x_ori**2 + new_y_ori**2) + 0.2
# new_z_ori = new_surface_z * 1.0
# new_node = np.array([new_x_ori, new_y_ori, new_z_ori])
# sub_nodes.append(new_node)  # 添加到子节点列表

# all_nodes = [center_node] + sub_nodes

# # ================= 6. 绘制节点与3D箭头 =================
# # 绘制所有子节点的箭头（包含新节点）
# for i, sub_node in enumerate(sub_nodes):
#     start_point = sub_node
#     end_point = center_node
    
#     vec = end_point - start_point
#     dist = np.linalg.norm(vec)
#     if dist < START_OFFSET + END_OFFSET:
#         continue
#     unit_vec = vec / dist      
    
#     new_start = start_point + unit_vec * START_OFFSET
#     actual_end_offset = min(END_OFFSET, dist * 0.4) 
#     new_end = end_point - unit_vec * actual_end_offset
    
#     arrow = Arrow3D(
#         [new_start[0], new_end[0]], 
#         [new_start[1], new_end[1]], 
#         [new_start[2], new_end[2]], 
#         mutation_scale=2,
#         lw=0.5, 
#         arrowstyle="-|>", 
#         color=ARROW_COLOR, 
#         zorder=20
#     )
#     ax.add_artist(arrow)

# # 绘制中心点
# ax.scatter3D(center_node[0], center_node[1], center_node[2], 
#              color="#CCE4FF", s=20, edgecolors='black', linewidth=0.5, marker='^', zorder=30)

# # 绘制子节点（对新节点单独设置红色）
# for i, node in enumerate(sub_nodes):
#     if i == len(sub_nodes)-1:  # 最后一个是新增的红色节点
#         ax.scatter3D(node[0], node[1], node[2], 
#                      color="#FF0000", s=10, edgecolors='black', linewidth=0.5, zorder=30)
#     else:
#         ax.scatter3D(node[0], node[1], node[2], 
#                      color="#CCE4FF", s=10, edgecolors='black', linewidth=0.5, zorder=30)

# # ================= 7. 调整视角与样式 =================
# ax.view_init(elev=25, azim=120)
# ax.set_zlim(1, 4.0)
# ax.set_box_aspect((1, 1, 0.6))
# ax.dist = 5.8
# ax.draw(renderer=fig.canvas.get_renderer())

# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow2.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow2.png"), bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)
# plt.show()


# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.colors as mcolors
# from matplotlib.patches import FancyArrowPatch
# from mpl_toolkits.mplot3d import proj3d

# script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
# save_dir = os.path.join(script_dir, "imgs")
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)

# # ================= 1. 自定义3D箭头类 =================
# class Arrow3D(FancyArrowPatch):
#     def __init__(self, xs, ys, zs, *args, **kwargs):
#         target_color = kwargs.pop('color', 'black')
#         if 'facecolor' not in kwargs:
#             kwargs['facecolor'] = target_color
#         if 'edgecolor' not in kwargs:
#             kwargs['edgecolor'] = target_color
#         kwargs['fill'] = True  # 箭头实心
#         if 'lw' not in kwargs and 'linewidth' not in kwargs:
#             kwargs['lw'] = 0.8
#         super().__init__((0, 0), (0, 0), *args, **kwargs)
#         self._verts3d = xs, ys, zs

#     def do_3d_projection(self, renderer=None):
#         xs3d, ys3d, zs3d = self._verts3d
#         xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
#         self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
#         return np.min(zs)

# # ================= 2. 核心参数与旋转计算 =================
# COLOR_SURF_START = "#151515"  
# COLOR_SURF_END = "#FAFAFA"    
# NODE_COLORS = ["#C71585", "#87CEEB", "#FFA500", "#98FB98", "#9370DB"]
# ARROW_COLOR = "#000000"

# START_OFFSET = 0.25
# END_OFFSET = 0.35
# # 新增：新点的箭头偏移（与原点区分，避免重叠）
# START_OFFSET_NEW = 0.3
# END_OFFSET_NEW = 0.4

# STOP_RATIO = 0.1  
# NODE_OFFSET = 0.12
# xy_limit = 3.0
# line_width = 0.05

# # --- 120°方位角下的视觉象限坐标计算 ---
# azim_angle = 120  
# theta = np.radians(azim_angle)
# cos_theta = np.cos(theta)
# sin_theta = np.sin(theta)

# # 原4个点的视觉象限（内侧）
# visual_quadrants = [
#     (-1.3, 1.3),   # 视觉左上
#     (1.3, 1.3),    # 视觉右上
#     (-1.5, -12.5/6),  # 视觉左下
#     (1.5, -2.5/6)    # 视觉右下
# ]

# # 新增4个点的视觉象限（外侧，与原点同方向但偏移0.5，避免重叠）
# visual_quadrants_new = [
#     (-1.8, 1.8),   # 新左上（原左上外围）
#     (1.8, 1.8),    # 新右上（原右上外围）
#     (-2.0, -2.0),  # 新左下（原左下外围）
#     (2.0, -2.0)    # 新右下（原右下外围）
# ]

# # 计算原4个点的原始坐标
# original_coords = []
# for x_vis, y_vis in visual_quadrants:
#     x_ori = x_vis * cos_theta + y_vis * sin_theta
#     y_ori = -x_vis * sin_theta + y_vis * cos_theta
#     original_coords.append((x_ori, y_ori))

# # 计算新4个点的原始坐标
# original_coords_new = []
# for x_vis, y_vis in visual_quadrants_new:
#     x_ori = x_vis * cos_theta + y_vis * sin_theta
#     y_ori = -x_vis * sin_theta + y_vis * cos_theta
#     original_coords_new.append((x_ori, y_ori))

# # ================= 3. 生成洛伦兹双曲面上叶 =================
# x = np.linspace(-3, 3, 120)
# y = np.linspace(-3, 3, 120)
# X, Y = np.meshgrid(x, y)
# scaling_factor = 1.0  
# Z = np.sqrt(1 + (scaling_factor * X)**2 + (scaling_factor * Y)**2)

# # ================= 4. 初始化画布与绘图 =================
# fig = plt.figure(figsize=(1.4, 1.3), dpi=300)
# fig.patch.set_alpha(0.0)
# ax = fig.add_subplot(111, projection='3d')
# ax.patch.set_alpha(0.0)
# ax.zaxis.set_rotate_label(False)
# ax.xaxis.set_rotate_label(False)
# ax.yaxis.set_rotate_label(False)
# ax.set_axis_off()

# custom_cmap = mcolors.LinearSegmentedColormap.from_list(
#     "surf_gradient", [COLOR_SURF_START, COLOR_SURF_END]
# )
# ax.plot_surface(
#     X, Y, Z,
#     cmap=custom_cmap,
#     edgecolor=(0, 0, 0, 0.15),
#     rstride=4, cstride=4,
#     alpha=0.4,
#     shade=False,
#     linewidth=line_width,
#     antialiased=True,
#     zorder=1
# )

# # ================= 5. 生成节点坐标 =================
# # 中心节点
# center_node = np.array([0, 0, np.sqrt(1 + 0**2 + 0**2)])

# # 原4个点的坐标
# sub_nodes = []
# for i, (x_ori, y_ori) in enumerate(original_coords):
#     surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.2
#     z_ori = surface_z * 1.0
#     sub_nodes.append(np.array([x_ori, y_ori, z_ori]))

# # 新4个点的坐标
# sub_nodes_new = []
# for i, (x_ori, y_ori) in enumerate(original_coords_new):
#     surface_z = np.sqrt(1 + x_ori**2 + y_ori**2) + 0.2
#     z_ori = surface_z * 1.0
#     sub_nodes_new.append(np.array([x_ori, y_ori, z_ori]))

# all_nodes = [center_node] + sub_nodes + sub_nodes_new

# # ================= 6. 绘制节点与3D箭头 =================
# # 绘制原4个点的箭头
# for i, sub_node in enumerate(sub_nodes):
#     start_point = sub_node
#     end_point = center_node
    
#     vec = end_point - start_point
#     dist = np.linalg.norm(vec)
#     if dist < START_OFFSET + END_OFFSET:
#         continue
#     unit_vec = vec / dist      
    
#     new_start = start_point + unit_vec * START_OFFSET
#     actual_end_offset = min(END_OFFSET, dist * 0.4) 
#     new_end = end_point - unit_vec * actual_end_offset
    
#     arrow = Arrow3D(
#         [new_start[0], new_end[0]], 
#         [new_start[1], new_end[1]], 
#         [new_start[2], new_end[2]], 
#         mutation_scale=4,
#         lw=0.5, 
#         arrowstyle="-|>", 
#         color=ARROW_COLOR, 
#         zorder=20
#     )
#     ax.add_artist(arrow)

# # 绘制新4个点的箭头（偏移与原点不同，避免重叠）
# for i, sub_node in enumerate(sub_nodes_new):
#     start_point = sub_node
#     end_point = center_node
    
#     vec = end_point - start_point
#     dist = np.linalg.norm(vec)
#     if dist < START_OFFSET_NEW + END_OFFSET_NEW:
#         continue
#     unit_vec = vec / dist      
    
#     new_start = start_point + unit_vec * START_OFFSET_NEW
#     actual_end_offset = min(END_OFFSET_NEW, dist * 0.4) 
#     new_end = end_point - unit_vec * actual_end_offset
    
#     arrow = Arrow3D(
#         [new_start[0], new_end[0]], 
#         [new_start[1], new_end[1]], 
#         [new_start[2], new_end[2]], 
#         mutation_scale=4,
#         lw=0.5, 
#         arrowstyle="-|>", 
#         color=ARROW_COLOR, 
#         zorder=20
#     )
#     ax.add_artist(arrow)

# # 绘制中心点
# ax.scatter3D(center_node[0], center_node[1], center_node[2], 
#              color="#CCE4FF", s=20, edgecolors='black', linewidth=0.5, marker='^', zorder=30)

# # 绘制原4个点
# for i, node in enumerate(sub_nodes):
#     ax.scatter3D(node[0], node[1], node[2], 
#                  color="#CCE4FF", s=20, edgecolors='black', linewidth=0.5, zorder=30)

# # 绘制新4个点（颜色按要求设置：左上/右上灰蓝，左下三文鱼红，右下明黄）
# new_node_colors = ["#80C8C8", "#80C8C8", "#FA7F6F", "#FDD865"]
# for i, node in enumerate(sub_nodes_new):
#     ax.scatter3D(node[0], node[1], node[2], 
#                  color=new_node_colors[i], s=20, edgecolors='black', linewidth=0.5, zorder=30)

# # ================= 7. 调整视角与样式 =================
# ax.view_init(elev=25, azim=120)
# ax.set_zlim(1, 4.0)  # 扩大z范围，容纳外围新点
# ax.set_box_aspect((1, 1, 0.6))
# ax.dist = 5.8
# ax.draw(renderer=fig.canvas.get_renderer())

# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow2.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
# plt.savefig(os.path.join(save_dir, "hyperbolic_graph_arrow2.png"), bbox_inches="tight", pad_inches=0, transparent=True, dpi=600)
# plt.show()