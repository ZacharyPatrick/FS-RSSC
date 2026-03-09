# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # ==========================================
# # 1. 路径与环境设置 (Robust Path Handling)
# # ==========================================
# # 获取当前脚本所在目录，确保图片保存路径永远正确
# script_dir = os.path.dirname(os.path.abspath(__file__))
# save_dir = os.path.join(script_dir, "imgs")

# # 检查并创建保存目录
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#     print(f"文件夹 '{save_dir}' 不存在，已自动创建。")

# # ==========================================
# # 2. 生成洛伦兹曲面数据 (Lorentz Model Data)
# # ==========================================
# # 公式: -t^2 + x^2 + y^2 = -1  =>  t = sqrt(1 + x^2 + y^2)
# # 我们用 Z 轴代表 t 轴 (时间/垂直轴)

# # 设置网格密度 (TGRS 偏好适中的密度，太密会变成黑块，太疏看不出曲率)
# # 半径范围 r: 0 (底部) 到 2.3 (碗口)，2.3 是一个经验值，既能显出弯曲，又不会太夸张
# r = np.linspace(0, 2.3, 25)  
# theta = np.linspace(0, 2 * np.pi, 60)
# r, theta = np.meshgrid(r, theta)

# # 转换为笛卡尔坐标
# X = r * np.cos(theta)
# Y = r * np.sin(theta)
# Z = np.sqrt(1 + X**2 + Y**2) 

# # ==========================================
# # 3. 绘图设置 (TGRS Aesthetic Config)
# # ==========================================
# fig = plt.figure(figsize=(10, 8))
# # 背景设置为纯白，方便论文排版
# fig.patch.set_facecolor('white') 
# ax = fig.add_subplot(111, projection='3d')

# # -------------------------------------------------------
# # 配色方案: [HHAP-Net 参考风格 - Tech Blue Hierarchy]
# # -------------------------------------------------------
# # TGRS 论文常用这种冷色调来表示"高维特征空间"
# # 面颜色: AliceBlue (#F0F8FF) 或 LightCyan (#E0FFFF) - 极淡，像空气一样
# # 线颜色: SlateGray (#708090) 或 SteelBlue (#4682B4) - 结构感强
# # -------------------------------------------------------

# SURFACE_COLOR = '#E0FFFF'   # 淡青色 (Light Cyan)，干净通透
# EDGE_COLOR    = '#5F9EA0'   # 军校蓝 (Cadet Blue)，作为网格线，专业且清晰
# ALPHA_VAL     = 0.15        # 透明度 (0.1-0.2 最佳)，保证不遮挡内部数据点

# # 绘制曲面 (The Bowl)
# surf = ax.plot_surface(X, Y, Z, 
#                        color=SURFACE_COLOR, 
#                        edgecolor=EDGE_COLOR,
#                        linewidth=0.5,       # 线条细一点，精致
#                        alpha=ALPHA_VAL,     # 高透明度
#                        rstride=1, cstride=1,# 网格采样步长
#                        antialiased=True,
#                        shade=False)         # 关闭自动光影，保持扁平化工程风

# # ==========================================
# # 4. 增强细节 (Optional Polish)
# # ==========================================
# # 为了让"碗"的形状更清晰，我们可以单独加粗"碗口"的边缘线
# # 获取最外圈的数据
# rim_x = X[-1, :] # 这种切片方式取决于meshgrid的生成顺序，如果不对可以尝试 X[:, -1]
# rim_y = Y[-1, :]
# rim_z = Z[-1, :]
# # 如果上述切片是径向的，我们需要提取 r=max 时的那一圈
# # 重新生成一圈纯圆的数据作为碗口
# theta_rim = np.linspace(0, 2*np.pi, 100)
# rim_x = 2.3 * np.cos(theta_rim)
# rim_y = 2.3 * np.sin(theta_rim)
# rim_z = np.sqrt(1 + 2.3**2) * np.ones_like(theta_rim)

# # 绘制碗口轮廓
# ax.plot(rim_x, rim_y, rim_z, color=EDGE_COLOR, linewidth=1.5, alpha=0.6)

# # ==========================================
# # 5. 视角与输出 (View & Save)
# # ==========================================
# # 移除坐标轴 (TGRS 示意图通常不需要坐标刻度)
# ax.set_axis_off()

# # 调整视角 (Elev: 仰角, Azim: 方位角)
# # elev=20, azim=30 是经典的"侧俯视"角度，能看清碗底，也能看清碗壁
# ax.view_init(elev=18, azim=25)

# # 设置紧凑布局
# plt.tight_layout()

# # 保存路径
# svg_path = os.path.join(save_dir, "HHAP_Style_Lorentz.svg")
# png_path = os.path.join(save_dir, "HHAP_Style_Lorentz.png")

# # 导出矢量图 (SVG) - 推荐用于 PPT 组合
# plt.savefig(svg_path, transparent=True, format="svg", bbox_inches='tight', pad_inches=0)

# # 导出高清位图 (PNG) - 600 DPI，印刷级清晰度
# plt.savefig(png_path, transparent=True, dpi=600, bbox_inches='tight', pad_inches=0)

# print(f"✅ 生成成功！")
# print(f"矢量图路径: {svg_path}")
# print(f"高清图路径: {png_path}")
# print("您可以直接将 SVG 文件拖入 PPT 中进行后续的打点和连线操作。")

# plt.show()


# import os
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_hyperbolic_surface():
#     # 获取当前脚本所在目录，确保图片保存路径永远正确
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     save_dir = os.path.join(script_dir, "imgs")

#     # 检查并创建保存目录
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#         print(f"文件夹 '{save_dir}' 不存在，已自动创建。")

#     # 1. 创建画布
#     # figsize 控制图片比例，dpi 控制清晰度
#     fig = plt.figure(figsize=(10, 8), dpi=300)
#     ax = fig.add_subplot(111, projection='3d')

#     # 2. 生成双曲面数据 (Lorentz Model / Hyperboloid)
#     # 洛伦兹模型上叶方程: z = sqrt(1 + x^2 + y^2)
#     limit = 2.5  # 控制曲面开口的大小，数值越大开口越广
#     resolution = 100 # 数据点的密度
#     x = np.linspace(-limit, limit, resolution)
#     y = np.linspace(-limit, limit, resolution)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     # 3. 绘制曲面
#     # rstride 和 cstride 决定了网格线的疏密程度，值越大网格越稀疏
#     # alpha 控制透明度，0.6 左右能透出背后的网格，增加立体感
#     surf = ax.plot_surface(X, Y, Z, 
#                            color='whitesmoke',   # 曲面底色，whitesmoke 或 lightgray 很适合论文
#                            edgecolor='darkgray', # 网格线颜色
#                            alpha=0.6,            # 透明度
#                            rstride=5,            # 行跨度 (控制网格密度)
#                            cstride=5,            # 列跨度 (控制网格密度)
#                            linewidth=0.3,        # 网格线粗细
#                            antialiased=True,     # 抗锯齿，线条更平滑
#                            shade=True)           # 开启光照阴影，更有体积感

#     # 4. 调整视角和美化
#     # 移除坐标轴背景、刻度和边框，使其成为纯净的素材图
#     ax.set_axis_off()
    
#     # 设置视角：elev 是仰角，azim 是方位角
#     # 25度和45度通常能获得较好的立体展示效果
#     ax.view_init(elev=25, azim=45)

#     # 可选：调整Z轴范围，剪裁掉过高的部分，聚焦于底部“碗”状区域
#     ax.set_zlim(1, 3.5)

#     # 5. 保存或显示
#     # 保存为透明背景的 PDF 或 PNG，方便后续叠加内容
#     # 如果要导入 AI 编辑，建议保存为 .pdf 或 .svg
#     plt.tight_layout()
#     svg_path = os.path.join(save_dir, "hyperbolic_surface_base.svg")
#     png_path = os.path.join(save_dir, "hyperbolic_surface_base.png")

#     plt.savefig(svg_path, transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(png_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# if __name__ == "__main__":
#     plot_hyperbolic_surface()


# import os
# import matplotlib.pyplot as plt
# import numpy as np

# def plot_custom_hyperbolic():
#     # 获取当前脚本所在目录
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     save_dir = os.path.join(script_dir, "imgs")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ================= 自定义参数区 (修改这里来改变效果) =================
    
#     # 1. 视角控制
#     view_elev = 25   
#     view_azim = 120   
    
#     # 2. 形状控制
#     xy_limit = 3.0   
    
#     # 3. 网格样式 (【修改点：大幅优化网格质感】)
#     # 将密度从 4 改为 8，网格线会变少，看起来更干净
#     grid_density = 8 
    
#     # 线宽从 0.2 改为 0.05，极细，只有在放大时才看清，不干扰视线
#     line_width = 0.05 
    
#     # 4. 颜色与材质 (【修改点：增强体积感】)
#     # 不再用 whitesmoke，改用 hex 色值 #D3D3D3 (LightGray)，颜色更深一点
#     surface_color = '#D3D3D3' 
    
#     # 【关键修改】：网格线颜色。
#     # 使用 RGBA 格式：(R, G, B, Alpha)。
#     # (0,0,0, 0.1) 代表纯黑色，但透明度只有 10%。
#     # 这种线条会极其柔和，完美解决“喧宾夺主”的问题。
#     grid_edge_color = (0, 0, 0, 0.1) 
    
#     # 曲面本身的透明度，保持 0.7 或 0.8，既有实体感又能透视
#     surface_opacity = 0.8    
    
#     # ===================================================================

#     fig = plt.figure(figsize=(10, 8), dpi=300) # 提高 dpi 预览更清晰
#     ax = fig.add_subplot(111, projection='3d')

#     # 生成数据
#     x = np.linspace(-xy_limit, xy_limit, 120) # 稍微提高采样率保证平滑
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     # 绘制曲面
#     ax.plot_surface(X, Y, Z, 
#                     color=surface_color,
#                     edgecolor=grid_edge_color, # 使用新的半透明网格色
#                     alpha=surface_opacity,
#                     rstride=grid_density,  
#                     cstride=grid_density,  
#                     linewidth=line_width,  # 使用极细线宽
#                     antialiased=True,
#                     shade=True)            # 开启光照阴影

#     # 移除背景刻度
#     ax.set_axis_off()
    
#     # 应用视角
#     ax.view_init(elev=view_elev, azim=view_azim)
    
#     # 裁剪
#     ax.set_zlim(1, 3.5)
#     ax.set_box_aspect((1, 1, 0.6)) 

#     print(f"生成完毕 -> 视角: {view_elev}/{view_azim}, 网格密度: {grid_density}")
    
#     plt.tight_layout()
#     svg_path = os.path.join(save_dir, "hyperbolic_surface_optimized.svg")
#     png_path = os.path.join(save_dir, "hyperbolic_surface_optimized.png")
    
#     # 保存图片
#     plt.savefig(svg_path, transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(png_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# if __name__ == "__main__":
#     plot_custom_hyperbolic()


# import os
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# def plot_gradient_hyperbolic():
#     # 获取当前脚本所在目录
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     save_dir = os.path.join(script_dir, "imgs")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ================= 自定义参数区 =================
    
#     view_elev = 25   
#     view_azim = 120   
#     xy_limit = 3.0   
    
#     # 网格样式 (保持之前的优化)
#     grid_density = 4 
#     line_width = 0.05 
#     grid_edge_color = (0, 0, 0, 0.15) # 稍微加深一点点网格，因为底色变了，需保持对比度
    
#     # 【核心修改 1】：定义渐变色 (Colormap)
#     # 我们创建一个从 "深灰 (#696969)" 到 "浅烟白 (whitesmoke)" 的渐变
#     # 你可以调整这两个 hex 值来控制“深浅”程度
#     # 底部颜色 (Bottom): #808080 (灰色) -> 建议不要太黑，否则像黑洞
#     # 顶部颜色 (Top): #F5F5F5 (接近白色的浅灰)
#     colors = ["#151515", "#FAFAFA"]    # 论文标准色 
#     cmap_name = "hyperbolic_fade"
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
    
#     # 透明度
#     surface_opacity = 0.85 
    
#     # =================================================

#     fig = plt.figure(figsize=(10, 8), dpi=300)
#     ax = fig.add_subplot(111, projection='3d')

#     # 生成数据
#     x = np.linspace(-xy_limit, xy_limit, 120)
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     # 【核心修改 2】：绘制曲面
#     surf = ax.plot_surface(X, Y, Z, 
#                            cmap=custom_cmap,    # 使用自定义的“深->浅”渐变
#                            edgecolor=grid_edge_color,
#                            alpha=surface_opacity,
#                            rstride=grid_density,  
#                            cstride=grid_density,  
#                            linewidth=line_width,
#                            antialiased=True,
#                            shade=False)         # 【关键】关闭光照！由 cmap 决定颜色，而非光照角度

#     # 移除背景
#     ax.set_axis_off()
    
#     # 应用视角
#     ax.view_init(elev=view_elev, azim=view_azim)
    
#     # 裁剪
#     ax.set_zlim(1, 3.5)
#     ax.set_box_aspect((1, 1, 0.6)) 

#     print(f"生成完毕 -> 渐变模式: Bottom({colors[0]}) -> Top({colors[1]})")
    
#     plt.tight_layout()
#     svg_path = os.path.join(save_dir, "hyperbolic_gradient.svg")
#     png_path = os.path.join(save_dir, "hyperbolic_gradient.png")
    
#     # 保存图片
#     plt.savefig(svg_path, transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(png_path, transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# if __name__ == "__main__":
#     plot_gradient_hyperbolic()


# import os
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # ================= 工具函数：生成贴合曲面的数据簇 =================
# def generate_cluster(center_x, center_y, num_points=5, spread=0.15, random_seed=None):
#     """
#     生成一簇围绕 (center_x, center_y) 的点
#     spread: 调小这个值，让点更紧凑，看起来更像一个“类”
#     """
#     if random_seed:
#         np.random.seed(random_seed)
    
#     # 1. 在 x, y 平面上生成围绕中心的随机散点
#     x = np.random.normal(center_x, spread, num_points)
#     y = np.random.normal(center_y, spread, num_points)
    
#     # 2. 计算 z 值，使其贴合洛伦兹曲面
#     # z = sqrt(1 + x^2 + y^2)
#     # +0.05 是为了让点稍微浮起一点点，避免与网格线打架，但在 zorder=10 的加持下，这个值主要保证物理位置正确
#     # z = np.sqrt(1 + x**2 + y**2) + 0.05
#     z = np.sqrt(1 + x**2 + y**2) + 2.0
    
#     return x, y, z

# def plot_hyperbolic_final_correction():
#     # 获取当前脚本所在目录
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     save_dir = os.path.join(script_dir, "imgs")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ================= 1. 底座参数 =================
#     view_elev = 25   
#     view_azim = 120   # 保持你喜欢的角度
#     xy_limit = 3.0   
#     grid_density = 4      
#     line_width = 0.05 
#     grid_edge_color = (0, 0, 0, 0.15) 
    
#     colors = ["#151515", "#FAFAFA"] 
#     cmap_name = "hyperbolic_fade"
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)
#     surface_opacity = 0.85 
    
#     fig = plt.figure(figsize=(10, 8), dpi=300)
#     ax = fig.add_subplot(111, projection='3d')

#     # ================= 2. 绘制底座 =================
#     x = np.linspace(-xy_limit, xy_limit, 120)
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     # zorder=1: 确保曲面在最底层
#     ax.plot_surface(X, Y, Z, 
#                     cmap=custom_cmap,
#                     edgecolor=grid_edge_color,
#                     alpha=surface_opacity,
#                     rstride=grid_density,  
#                     cstride=grid_density,  
#                     linewidth=line_width,
#                     antialiased=True,
#                     shade=False,
#                     zorder=1) 

#     # ================= 3. 绘制数据点 (关键位置调整) =================
    
#     # 【位置策略调整】：
#     # 不要把点放在 x=1.5 或 -1.8 这种边缘，因为 view_azim=120 时那里是视线死角。
#     # 我们把点集中在 (0,0) 附近的 "碗底区域"。
    
#     # --- 类别 A (黄色，In-class) ---
#     # 放在正中心偏左一点
#     cx_a, cy_a = 0.2, -0.2
#     feat_x_a, feat_y_a, feat_z_a = generate_cluster(cx_a, cy_a, num_points=8, spread=0.15, random_seed=42)
#     # 原型 A
#     proto_z_a = np.sqrt(1 + cx_a**2 + cy_a**2) + 0.05

#     # --- 类别 B (红色，Out-of-class) ---
#     # 放在右上方，但不要太远
#     cx_b, cy_b = -0.8, 0.6
#     feat_x_b, feat_y_b, feat_z_b = generate_cluster(cx_b, cy_b, num_points=6, spread=0.15, random_seed=1)
#     proto_z_b = np.sqrt(1 + cx_b**2 + cy_b**2) + 0.05

#     # --- 类别 C (蓝色，Out-of-class) ---
#     # 放在左上方
#     cx_c, cy_c = 0.7, 0.7
#     feat_x_c, feat_y_c, feat_z_c = generate_cluster(cx_c, cy_c, num_points=6, spread=0.15, random_seed=2)
#     proto_z_c = np.sqrt(1 + cx_c**2 + cy_c**2) + 0.05

#     # ================= 4. 强力绘图 (Z-order Magic) =================
#     # 关键参数：
#     # 1. zorder=10: 强制让点显示在曲面上方，彻底解决“看不见”的问题
#     # 2. depthshade=False: 关闭深度阴影，让颜色更纯正（可选，如果你觉得颜色太暗就关掉它）
#     # 3. edgecolors='white', linewidth=0.8: 加粗白边，增强对比度
    
#     # 画特征点
#     ax.scatter(feat_x_a, feat_y_a, feat_z_a, s=100, c='#FFC107', marker='o', 
#                edgecolors='white', linewidth=0.8, zorder=10, label='Class A')
    
#     ax.scatter(feat_x_b, feat_y_b, feat_z_b, s=100, c='#E53935', marker='o', 
#                edgecolors='white', linewidth=0.8, zorder=10, label='Class B')

#     ax.scatter(feat_x_c, feat_y_c, feat_z_c, s=100, c='#1E88E5', marker='o', 
#                edgecolors='white', linewidth=0.8, zorder=10, label='Class C')


#     # 画原型 (Prototype) - 星星更大
#     ax.scatter([cx_a], [cy_a], [proto_z_a], s=350, c='#FFC107', marker='*', 
#                edgecolors='black', linewidth=0.5, zorder=20, label='Proto A')
    
#     ax.scatter([cx_b], [cy_b], [proto_z_b], s=350, c='#E53935', marker='*', 
#                edgecolors='black', linewidth=0.5, zorder=20, label='Proto B')

#     ax.scatter([cx_c], [cy_c], [proto_z_c], s=350, c='#1E88E5', marker='*', 
#                edgecolors='black', linewidth=0.5, zorder=20, label='Proto C')


#     # ================= 5. 收尾 =================
#     ax.set_axis_off()
#     ax.view_init(elev=view_elev, azim=view_azim)
#     ax.set_zlim(1, 3.5)
#     ax.set_box_aspect((1, 1, 0.6)) 

#     print(f"修正完毕 -> 点已强制前置显示 (zorder=10)，并移向中心")
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "hyperbolic_final.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(save_dir, "hyperbolic_final.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# if __name__ == "__main__":
#     plot_hyperbolic_final_correction()


# import os
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # ================= 工具函数：生成碗内点（核心修改：增大z轴偏移） =================
# def generate_cluster(center_x, center_y, num_points=4, spread=0.15, random_seed=None):
#     """
#     生成一簇在洛伦兹曲面内部的点（彻底避免遮挡）
#     关键修改：z轴偏移从0.3增大到0.8，确保视觉上有明显高度差
#     """
#     if random_seed:
#         np.random.seed(random_seed)
#     # 1. 生成平面散点（扩大spread，让点分布更分散，避免重叠）
#     x = np.random.normal(center_x, spread * 4.0, num_points)
#     y = np.random.normal(center_y, spread * 2.0, num_points)
#     # 2. 核心：z值远小于曲面（偏移0.8，视觉上明显在内部）
#     z = np.sqrt(1 + x**2 + y**2) + 1.0
#     return x, y, z

# def plot_hyperbolic_final_correction():
#     # 保存路径
#     script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
#     save_dir = os.path.join(script_dir, "imgs")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ================= 1. 核心参数（全量优化） =================
#     view_elev = 25   # 提高仰角，减少碗壁遮挡
#     view_azim = 120   # 改为正侧视角（90°），彻底避免120°的死角遮挡
#     xy_limit = 3.0   # 缩小曲面范围，让点更集中在视野中心
#     grid_density = 4 # 减小面片尺寸（rstride/cstride用8），减少面片遮挡
#     line_width = 0.05 # 网格线极细，避免遮挡
#     grid_edge_color = (0, 0, 0, 0.15) # 网格线几乎透明
#     surface_opacity = 0.4 # 降低曲面透明度，让内部点透出来
#     point_size = 150 # 增大点的尺寸，视觉上更突出
#     proto_size = 400 # 原型点尺寸增大

#     # 配色：增强对比（深色曲面+亮色点）
#     colors = ["#2C2C2C", "#F0F0F0"] 
#     cmap_name = "hyperbolic_fade"
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

#     # ================= 2. 绘图初始化（关键：关闭深度缓冲对曲面的优先级） =================
#     fig = plt.figure(figsize=(10, 8), dpi=300)
#     ax = fig.add_subplot(111, projection='3d')
#     # 关闭3D网格的深度测试（核心！让曲面不遮挡后画的点）
#     ax.zaxis.set_rotate_label(False)
#     ax.xaxis.set_rotate_label(False)
#     ax.yaxis.set_rotate_label(False)

#     # ================= 3. 绘制曲面（优先画，且降低遮挡性） =================
#     x = np.linspace(-xy_limit, xy_limit, 120)
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     # 曲面绘制参数：最小化遮挡
#     ax.plot_surface(X, Y, Z, 
#                     cmap=custom_cmap,
#                     edgecolor=grid_edge_color,
#                     alpha=surface_opacity,
#                     rstride=grid_density,  # 小面片，减少遮挡
#                     cstride=grid_density,
#                     linewidth=line_width,
#                     antialiased=True,
#                     shade=True, # 开启光影，增强曲面层次感，和点区分开
#                     zorder=1) # 曲面放最底层

#     # ================= 4. 绘制数据点（后画，且增强视觉对比） =================
#     # 调整点的位置：分散在视野中，避免重叠
#     # --- 类别 A（黄色，中心）---
#     cx_a, cy_a = 0.0, 0.0
#     feat_x_a, feat_y_a, feat_z_a = generate_cluster(cx_a, cy_a, num_points=4, spread=0.2, random_seed=2)
#     proto_z_a = np.sqrt(1 + cx_a**2 + cy_a**2) + 0.0

#     # --- 类别 B（红色，右侧）---
#     cx_b, cy_b = 0.8, -0.5
#     feat_x_b, feat_y_b, feat_z_b = generate_cluster(cx_b, cy_b, num_points=2, spread=0.3, random_seed=2)
#     # proto_z_b = np.sqrt(1 + cx_b**2 + cy_b**2) + 1.0

#     # --- 类别 C（蓝色，左侧）---
#     cx_c, cy_c = -0.8, 0.5
#     feat_x_c, feat_y_c, feat_z_c = generate_cluster(cx_c, cy_c, num_points=1, spread=0.15, random_seed=1)
#     # proto_z_c = np.sqrt(1 + cx_c**2 + cy_c**2) + 1.0

#     # ================= 5. 绘制点（核心：增强对比，后画优先） =================
#     # 特征点：深色边框+亮色填充，对比拉满
#     ax.scatter3D(feat_x_a, feat_y_a, feat_z_a, s=point_size, c='#FFC107', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class A')
#     ax.scatter3D(feat_x_b, feat_y_b, feat_z_b, s=point_size, c='#E53935', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
#     ax.scatter3D(feat_x_c, feat_y_c, feat_z_c, s=point_size, c='#1E88E5', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class C')

#     # 原型点：更大尺寸+黑色边框，视觉突出
#     ax.scatter3D([cx_a], [cy_a], [proto_z_a], s=proto_size, c='#FFC107', marker='*', 
#                edgecolors='#000000', linewidth=1.0, zorder=200, label='Proto A')
#     # ax.scatter3D([cx_b], [cy_b], [proto_z_b], s=proto_size, c='#E53935', marker='*', 
#     #            edgecolors='#000000', linewidth=1.0, zorder=200, label='Proto B')
#     # ax.scatter3D([cx_c], [cy_c], [proto_z_c], s=proto_size, c='#1E88E5', marker='*', 
#     #            edgecolors='#000000', linewidth=1.0, zorder=200, label='Proto C')

#     # ================= 6. 视角与坐标轴优化（消除压缩） =================
#     ax.set_axis_off() # 关闭坐标轴
#     ax.view_init(elev=view_elev, azim=view_azim)
#     ax.set_zlim(1, 3.5) # 调整z轴范围，放大点的高度差
#     ax.set_box_aspect((1, 1, 0.6)) # 取消z轴压缩！让点的高度差正常显示
#     # 强制刷新渲染顺序（确保点在最顶层）
#     ax.draw(renderer=fig.canvas.get_renderer())

#     # ================= 7. 保存与显示 =================
#     print("修复完成：点已清晰可见，无遮挡")
#     plt.tight_layout()
#     # 保存为矢量图+高清位图
#     plt.savefig(os.path.join(save_dir, "hyperbolic_final.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(save_dir, "hyperbolic_final.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# if __name__ == "__main__":
#     plot_hyperbolic_final_correction()


# 绘制从特征点指向原型点的虚线形式的直线箭头
# import os
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # ================= 工具函数：生成碗内点 =================
# def generate_cluster(center_x, center_y, num_points=4, spread=0.15, random_seed=None):
#     if random_seed:
#         np.random.seed(random_seed)
#     x = np.random.normal(center_x, spread * 4.0, num_points)
#     y = np.random.normal(center_y, spread * 2.0, num_points)
#     # 保持较大的Z轴偏移，确保点悬浮在碗内
#     z = np.sqrt(1 + x**2 + y**2) + 0.8 
#     return x, y, z

# # # ================= 新增：绘制防重叠连接线 =================
# # def draw_geodesic_arrows(ax, start_xs, start_ys, start_zs, end_x, end_y, end_z, color='black'):
# #     """
# #     绘制从特征点(start)指向原型(end)的虚线箭头
# #     核心策略：计算方向向量，让箭头停在距离终点一定距离的地方，防止重叠
# #     """
# #     stop_distance = 0.25 # 距离终点多远停下来（防重叠缓冲区）
# #     arrow_len = 0.07     # 箭头本身的长度
    
# #     for sx, sy, sz in zip(start_xs, start_ys, start_zs):
# #         # 1. 计算方向向量 (Start -> End)
# #         dx, dy, dz = end_x - sx, end_y - sy, end_z - sz
# #         full_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
# #         if full_dist < stop_distance: continue # 距离太近就不画了

# #         # 2. 计算归一化方向向量
# #         ux, uy, uz = dx/full_dist, dy/full_dist, dz/full_dist
        
# #         # 3. 计算“截断”后的终点坐标 (Line End)
# #         # 线条只画到 [总距离 - 停止距离] 的位置
# #         draw_dist = full_dist - stop_distance
# #         lx = sx + ux * draw_dist
# #         ly = sy + uy * draw_dist
# #         lz = sz + uz * draw_dist
        
# #         # 4. 绘制虚线 (Line)
# #         ax.plot([sx, lx], [sy, ly], [sz, lz], 
# #                 color=color, linestyle=(0, (3, 3)), # 自定义虚线样式：3pt线，3pt空
# #                 linewidth=0.7, alpha=1.0, zorder=90)
        
# #         # 5. 绘制箭头 (Arrow Head)
# #         # 在线条终点处画一个小小的箭头
# #         ax.quiver(lx, ly, lz, ux, uy, uz, 
# #                   length=arrow_len, normalize=True,
# #                   color=color, arrow_length_ratio=0.5, # 0.6让箭头看起来更尖
# #                   linewidth=0.5, alpha=1.0, zorder=90)

# # ================= 新增：绘制防重叠连接线 (修改版) =================
# def draw_geodesic_arrows(ax, start_xs, start_ys, start_zs, end_x, end_y, end_z, color='black'):
#     """
#     绘制从特征点(start)指向原型(end)的虚线箭头
#     【核心修改】：加粗线条，强制纯黑
#     """
#     stop_distance = 0.25 
#     arrow_len = 0.07     
    
#     # 【强制纯黑】：即便传入的color是黑，我们在这里硬编码确保万无一失
#     pure_black = '#000000'

#     for sx, sy, sz in zip(start_xs, start_ys, start_zs):
#         dx, dy, dz = end_x - sx, end_y - sy, end_z - sz
#         full_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
#         if full_dist < stop_distance: continue 

#         ux, uy, uz = dx/full_dist, dy/full_dist, dz/full_dist
        
#         draw_dist = full_dist - stop_distance
#         lx = sx + ux * draw_dist
#         ly = sy + uy * draw_dist
#         lz = sz + uz * draw_dist
        
#         # 1. 绘制虚线 (Line)
#         # 【修改点 1】：linewidth 从 0.7 提升到 1.5。
#         # 只有线条够粗，抗锯齿才不会把它模糊成灰色。
#         ax.plot([sx, lx], [sy, ly], [sz, lz], 
#                 color=pure_black, 
#                 linestyle=(0, (2, 2)), # 稍微调整虚线间隙，更紧凑
#                 linewidth=1.0,         # 加粗！
#                 alpha=1.0, 
#                 zorder=300)
        
#         # 2. 绘制箭头 (Arrow Head)
#         # 【修改点 2】：linewidth 从 0.5 提升到 1.2
#         # 并显式设置 edgecolor 和 facecolor
#         ax.quiver(lx, ly, lz, ux, uy, uz, 
#                   length=arrow_len, normalize=True,
#                   color=pure_black, 
#                   edgecolor=pure_black, # 强制边缘纯黑
#                   facecolor=pure_black, # 强制填充纯黑
#                   arrow_length_ratio=0.5, 
#                   linewidth=0.7,        # 加粗！
#                   alpha=1.0, 
#                   zorder=300)

# def plot_hyperbolic_final_with_arrows():
#     # 保存路径
#     script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
#     save_dir = os.path.join(script_dir, "imgs")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ================= 1. 场景参数 =================
#     view_elev = 25   
#     view_azim = 120   
#     xy_limit = 3.0   
#     grid_density = 4 
#     line_width = 0.05 
#     grid_edge_color = (0, 0, 0, 0.15) 
#     surface_opacity = 0.4 
#     point_size = 150 
#     proto_size = 400 

#     # colors = ["#2C2C2C", "#F0F0F0"]
#     # colors = ["#000000", "#F5F5F5"]
#     colors = ["#151515", "#FAFAFA"]    # 论文标准色 
#     cmap_name = "hyperbolic_fade"
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

#     # ================= 2. 绘图初始化 =================
#     fig = plt.figure(figsize=(10, 8), dpi=300)

#     # 【关键设置 1】：确保 Figure 背景是透明的
#     fig.patch.set_alpha(0.0)

#     ax = fig.add_subplot(111, projection='3d')

#     # 【关键设置 2】：确保 Axis 背景是透明的
#     ax.patch.set_alpha(0.0)
    
#     ax.zaxis.set_rotate_label(False)
#     ax.xaxis.set_rotate_label(False)
#     ax.yaxis.set_rotate_label(False)
#     ax.set_axis_off()

#     # ================= 3. 绘制曲面 =================
#     x = np.linspace(-xy_limit, xy_limit, 120)
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     ax.plot_surface(X, Y, Z, 
#                     cmap=custom_cmap,
#                     edgecolor=grid_edge_color,
#                     alpha=surface_opacity,
#                     rstride=grid_density,  
#                     cstride=grid_density,
#                     linewidth=line_width,
#                     antialiased=True,
#                     shade=False,
#                     zorder=1)

#     # ================= 4. 生成数据点 =================
#     # --- 类别 A（黄色，中心）---
#     cx_a, cy_a = 0.0, 0.0
#     feat_x_a, feat_y_a, feat_z_a = generate_cluster(cx_a, cy_a, num_points=4, spread=0.2, random_seed=2)
#     proto_z_a = np.sqrt(1 + cx_a**2 + cy_a**2) + 0.0 # 原型贴地一点

#     # --- 类别 B（红色，右侧）---
#     # 注意：这里稍微调整了B的坐标，让连接线角度更好看
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

#     # ================= 5. 绘制点 =================
#     # 特征点
#     ax.scatter3D(feat_x_a, feat_y_a, feat_z_a, s=point_size, c='#FFC107', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class A')
#     ax.scatter3D(feat_x_b_1, feat_y_b_1, feat_z_b_1, s=point_size, c='#E53935', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
#     ax.scatter3D(feat_x_b_2, feat_y_b_2, feat_z_b_2, s=point_size, c='#E53935', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
#     ax.scatter3D(feat_x_c, feat_y_c, feat_z_c, s=point_size, c='#1E88E5', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class C')
#     ax.scatter3D(feat_x_d, feat_y_d, feat_z_d, s=point_size, c='#43A047', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class D')

#     # 原型点
#     ax.scatter3D([cx_a], [cy_a], [proto_z_a], s=proto_size, c='#FFC107', marker='*', 
#                edgecolors='#000000', linewidth=1.0, zorder=200, label='Proto A')

#     # ================= 6. 绘制连接线 (关键步骤) =================
#     # 使用深灰色线条，比纯黑更柔和
#     link_color = '#000000'
    
#     # 绘制 A 类连接线
#     draw_geodesic_arrows(ax, feat_x_a, feat_y_a, feat_z_a, cx_a, cy_a, proto_z_a, color=link_color)
    
#     # B 类和 C 类虽然没有画出原型(被注释了)，但如果有需要，也可以画出指向虚拟原型的线
#     # 这里我们只画A类（因为您的原始代码只开启了A类的原型显示）
#     # 如果您想画出指向B类中心的线（即使不画星星）：
#     draw_geodesic_arrows(ax, feat_x_b_1, feat_y_b_1, feat_z_b_1, cx_a, cy_a, proto_z_a, color=link_color)
#     draw_geodesic_arrows(ax, feat_x_b_2, feat_y_b_2, feat_z_b_2, cx_a, cy_a, proto_z_a, color=link_color)
#     draw_geodesic_arrows(ax, feat_x_c, feat_y_c, feat_z_c, cx_a, cy_a, proto_z_a, color=link_color)
#     draw_geodesic_arrows(ax, feat_x_d, feat_y_d, feat_z_d, cx_a, cy_a, proto_z_a, color=link_color)

#     # ================= 7. 收尾 =================
#     ax.view_init(elev=view_elev, azim=view_azim)
#     ax.set_zlim(1, 3.5)
#     ax.set_box_aspect((1, 1, 0.6))
#     ax.draw(renderer=fig.canvas.get_renderer())

#     print("生成完毕：已添加防重叠虚线箭头")
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "hyperbolic_final_arrow.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(save_dir, "hyperbolic_final_arrow.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     plt.show()

# if __name__ == "__main__":
#     plot_hyperbolic_final_with_arrows()


# 绘制特征点到原型点间的测地线
# import os
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # ================= 工具函数：生成碗内点 =================
# def generate_cluster(center_x, center_y, num_points=4, spread=0.15, random_seed=None):
#     if random_seed:
#         np.random.seed(random_seed)
#     x = np.random.normal(center_x, spread * 4.0, num_points)
#     y = np.random.normal(center_y, spread * 2.0, num_points)
#     # 保持较大的Z轴偏移，确保点悬浮在碗内
#     z = np.sqrt(1 + x**2 + y**2) + 0.8 
#     return x, y, z

# # ================= 新增数学工具：双曲几何计算 =================

# def minkowski_dot(u, v):
#     """
#     计算闵可夫斯基内积。
#     在双叶双曲面模型 Z = sqrt(1 + X^2 + Y^2) 中，
#     度量签名通常为 (+, +, -) 或 (-, -, +)。
#     这里为了符合 Z^2 - X^2 - Y^2 = 1 的形式，我们使用: x1x2 + y1y2 - z1z2
#     """
#     return u[0]*v[0] + u[1]*v[1] - u[2]*v[2]

# def get_geodesic_path(start_p, end_p, num_points=50, stop_ratio=0.0):
#     """
#     计算两点之间的测地线路径点。
#     start_p, end_p: [x, y, z] numpy 数组
#     stop_ratio: 0.0~1.0, 提前多少比例停止（用于防重叠）
#     """
#     # 1. 计算闵可夫斯基内积
#     prod = minkowski_dot(start_p, end_p)
    
#     # 数值稳定性截断（防止浮点误差导致 prod > -1）
#     # 理论上两个不同的点内积应 <= -1
#     if prod > -1.0: 
#         prod = -1.0 - 1e-7
        
#     # 2. 计算双曲距离 d = arccosh(-<u,v>)
#     d = np.arccosh(-prod)
    
#     # 如果距离极小，直接返回起点
#     if d < 1e-6:
#         return np.tile(start_p, (num_points, 1)).T

#     # 3. 生成插值参数 t
#     # t 从 0 变化到 (1 - stop_ratio)，实现提前停止
#     # 注意：stop_ratio 是相对于总双曲距离 d 的比例
#     # 如果传入的是绝对距离 stop_dist，则 t_end = (d - stop_dist) / d
#     t_end = 1.0 - stop_ratio
#     t = np.linspace(0, t_end, num_points)
    
#     # 4. 测地线参数方程
#     # gamma(t) = (sinh((1-t)d)/sinh(d)) * start + (sinh(td)/sinh(d)) * end
#     sinh_d = np.sinh(d)
#     coeff_start = np.sinh((1 - t) * d) / sinh_d
#     coeff_end = np.sinh(t * d) / sinh_d
    
#     # 组合坐标: (num_points, 3)
#     # start_p[None, :] 广播成 (1, 3)
#     path = np.outer(coeff_start, start_p) + np.outer(coeff_end, end_p)
    
#     return path[:, 0], path[:, 1], path[:, 2] # 返回 x, y, z 数组

# # ================= 修改后：绘制双曲测地线箭头 =================
# def draw_geodesic_arrows(ax, start_xs, start_ys, start_zs, end_x, end_y, end_z, color='black'):
#     """
#     绘制从特征点(start)指向原型(end)的【测地线】虚线箭头
#     """
#     stop_distance_buffer = 0.35 # 停止距离（绝对值，非比例），防止覆盖原型
#     arrow_len = 0.15            # 箭头视觉长度
#     pure_black = '#000000'

#     end_vec = np.array([end_x, end_y, end_z])

#     for sx, sy, sz in zip(start_xs, start_ys, start_zs):
#         start_vec = np.array([sx, sy, sz])
        
#         # 1. 先计算一次距离，为了确定停止的比例
#         prod = minkowski_dot(start_vec, end_vec)
#         if prod > -1.0: prod = -1.0 - 1e-7
#         total_dist = np.arccosh(-prod)
        
#         if total_dist < stop_distance_buffer: continue # 距离太近不画

#         # 计算停止比例 (stop_ratio)
#         current_stop_ratio = stop_distance_buffer / total_dist
        
#         # 2. 获取测地线路径点 (Curve)
#         gx, gy, gz = get_geodesic_path(start_vec, end_vec, num_points=60, stop_ratio=current_stop_ratio)
        
#         # 3. 绘制弯曲的虚线
#         ax.plot(gx, gy, gz, 
#                 color=pure_black, 
#                 linestyle=(0, (3, 2)), # 虚线样式
#                 linewidth=1.2,         # 粗细
#                 alpha=0.9, 
#                 zorder=300)
        
#         # 4. 计算箭头方向 (Tangent)
#         # 既然是曲线，箭头的方向应该是曲线末端的切线方向
#         # 我们取路径最后两个点来近似切向量
#         if len(gx) >= 2:
#             dx = gx[-1] - gx[-2]
#             dy = gy[-1] - gy[-2]
#             dz = gz[-1] - gz[-2]
            
#             # 归一化切向量
#             norm = np.sqrt(dx**2 + dy**2 + dz**2)
#             if norm > 0:
#                 ux, uy, uz = dx/norm, dy/norm, dz/norm
#             else:
#                 ux, uy, uz = 0, 0, 0

#             # 5. 绘制箭头
#             ax.quiver(gx[-1], gy[-1], gz[-1], ux, uy, uz, 
#                       length=arrow_len, normalize=True,
#                       color=pure_black, 
#                       edgecolor=pure_black,
#                       facecolor=pure_black,
#                       arrow_length_ratio=0.4, 
#                       linewidth=0.8, 
#                       alpha=1.0, 
#                       zorder=301)

# def plot_hyperbolic_final_with_arrows():
#     # 保存路径
#     script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
#     save_dir = os.path.join(script_dir, "imgs")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ================= 1. 场景参数 =================
#     view_elev = 25   
#     view_azim = 120   
#     xy_limit = 3.0   
#     grid_density = 4 
#     line_width = 0.05 
#     grid_edge_color = (0, 0, 0, 0.15) 
#     surface_opacity = 0.4 
#     point_size = 150 
#     proto_size = 400 

#     colors = ["#151515", "#FAFAFA"]    
#     cmap_name = "hyperbolic_fade"
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

#     # ================= 2. 绘图初始化 =================
#     fig = plt.figure(figsize=(10, 8), dpi=300)
#     fig.patch.set_alpha(0.0)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.patch.set_alpha(0.0)
#     ax.zaxis.set_rotate_label(False)
#     ax.xaxis.set_rotate_label(False)
#     ax.yaxis.set_rotate_label(False)
#     ax.set_axis_off()

#     # ================= 3. 绘制曲面 =================
#     x = np.linspace(-xy_limit, xy_limit, 120)
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     ax.plot_surface(X, Y, Z, 
#                     cmap=custom_cmap,
#                     edgecolor=grid_edge_color,
#                     alpha=surface_opacity,
#                     rstride=grid_density,  
#                     cstride=grid_density,
#                     linewidth=line_width,
#                     antialiased=True,
#                     shade=False,
#                     zorder=1)

#     # ================= 4. 生成数据点 =================
#     # --- 类别 A（黄色，中心）---
#     cx_a, cy_a = 0.0, 0.0
#     feat_x_a, feat_y_a, feat_z_a = generate_cluster(cx_a, cy_a, num_points=4, spread=0.2, random_seed=2)
#     proto_z_a = np.sqrt(1 + cx_a**2 + cy_a**2) + 0.0 

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

#     # ================= 5. 绘制点 =================
#     ax.scatter3D(feat_x_a, feat_y_a, feat_z_a, s=point_size, c='#FFC107', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class A')
#     ax.scatter3D(feat_x_b_1, feat_y_b_1, feat_z_b_1, s=point_size, c='#E53935', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
#     ax.scatter3D(feat_x_b_2, feat_y_b_2, feat_z_b_2, s=point_size, c='#E53935', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
#     ax.scatter3D(feat_x_c, feat_y_c, feat_z_c, s=point_size, c='#1E88E5', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class C')
#     ax.scatter3D(feat_x_d, feat_y_d, feat_z_d, s=point_size, c='#43A047', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class D')

#     # 原型点
#     ax.scatter3D([cx_a], [cy_a], [proto_z_a], s=proto_size, c='#FFC107', marker='*', 
#                edgecolors='#000000', linewidth=1.0, zorder=200, label='Proto A')

#     # ================= 6. 绘制测地线连接 (核心修改) =================
#     link_color = '#000000'
    
#     # 绘制 A 类连接线 (测地线)
#     draw_geodesic_arrows(ax, feat_x_a, feat_y_a, feat_z_a, cx_a, cy_a, proto_z_a, color=link_color)
    
#     # 绘制其他类指向原型的线 (如果需要)
#     draw_geodesic_arrows(ax, feat_x_b_1, feat_y_b_1, feat_z_b_1, cx_a, cy_a, proto_z_a, color=link_color)
#     draw_geodesic_arrows(ax, feat_x_b_2, feat_y_b_2, feat_z_b_2, cx_a, cy_a, proto_z_a, color=link_color)
#     draw_geodesic_arrows(ax, feat_x_c, feat_y_c, feat_z_c, cx_a, cy_a, proto_z_a, color=link_color)
#     draw_geodesic_arrows(ax, feat_x_d, feat_y_d, feat_z_d, cx_a, cy_a, proto_z_a, color=link_color)

#     # ================= 7. 收尾 =================
#     ax.view_init(elev=view_elev, azim=view_azim)
#     ax.set_zlim(1, 3.5)
#     ax.set_box_aspect((1, 1, 0.6))
    
#     # 去除多余边距
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "hyperbolic_geodesic.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(save_dir, "hyperbolic_geodesic.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     print("生成完毕：已绘制精确的洛伦兹测地线连接")
#     plt.show()

# if __name__ == "__main__":
#     plot_hyperbolic_final_with_arrows()


# 绘制两两之间的测地线
# import os
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# import numpy as np

# # ================= 工具函数：生成碗内点 =================
# def generate_cluster(center_x, center_y, num_points=4, spread=0.15, random_seed=None):
#     if random_seed:
#         np.random.seed(random_seed)
#     x = np.random.normal(center_x, spread * 4.0, num_points)
#     y = np.random.normal(center_y, spread * 2.0, num_points)
#     # 保持较大的Z轴偏移，确保点悬浮在碗内
#     z = np.sqrt(1 + x**2 + y**2) + 0.8 
#     return x, y, z

# # ================= 新增数学工具：双曲几何计算 =================
# def minkowski_dot(u, v):
#     """
#     计算闵可夫斯基内积。
#     在双叶双曲面模型 Z = sqrt(1 + X^2 + Y^2) 中，
#     度量签名通常为 (+, +, -) 或 (-, -, +)。
#     这里为了符合 Z^2 - X^2 - Y^2 = 1 的形式，我们使用: x1x2 + y1y2 - z1z2
#     """
#     return u[0]*v[0] + u[1]*v[1] - u[2]*v[2]

# def get_geodesic_path(start_p, end_p, num_points=50, stop_ratio=0.0):
#     """
#     计算两点之间的测地线路径点。
#     start_p, end_p: [x, y, z] numpy 数组
#     stop_ratio: 0.0~1.0, 提前多少比例停止（用于防重叠）
#     """
#     # 1. 计算闵可夫斯基内积
#     prod = minkowski_dot(start_p, end_p)
    
#     # 数值稳定性截断（防止浮点误差导致 prod > -1）
#     # 理论上两个不同的点内积应 <= -1
#     if prod > -1.0: 
#         prod = -1.0 - 1e-7
        
#     # 2. 计算双曲距离 d = arccosh(-<u,v>)
#     d = np.arccosh(-prod)
    
#     # 如果距离极小，直接返回起点
#     if d < 1e-6:
#         return np.tile(start_p, (num_points, 1)).T

#     # 3. 生成插值参数 t
#     # t 从 0 变化到 (1 - stop_ratio)，实现提前停止
#     # 注意：stop_ratio 是相对于总双曲距离 d 的比例
#     # 如果传入的是绝对距离 stop_dist，则 t_end = (d - stop_dist) / d
#     t_end = 1.0 - stop_ratio
#     t = np.linspace(0, t_end, num_points)
    
#     # 4. 测地线参数方程
#     # gamma(t) = (sinh((1-t)d)/sinh(d)) * start + (sinh(td)/sinh(d)) * end
#     sinh_d = np.sinh(d)
#     coeff_start = np.sinh((1 - t) * d) / sinh_d
#     coeff_end = np.sinh(t * d) / sinh_d
    
#     # 组合坐标: (num_points, 3)
#     # start_p[None, :] 广播成 (1, 3)
#     path = np.outer(coeff_start, start_p) + np.outer(coeff_end, end_p)
    
#     return path[:, 0], path[:, 1], path[:, 2] # 返回 x, y, z 数组

# # ================= 修改后：绘制双曲测地线箭头 =================
# def draw_geodesic_arrows(ax, start_xs, start_ys, start_zs, end_x, end_y, end_z, color='black'):
#     """
#     绘制从特征点(start)指向原型(end)的【测地线】虚线箭头
#     """
#     stop_distance_buffer = 0.35 # 停止距离（绝对值，非比例），防止覆盖原型
#     arrow_len = 0.15            # 箭头视觉长度
#     pure_black = '#000000'

#     end_vec = np.array([end_x, end_y, end_z])

#     for sx, sy, sz in zip(start_xs, start_ys, start_zs):
#         start_vec = np.array([sx, sy, sz])
        
#         # 1. 先计算一次距离，为了确定停止的比例
#         prod = minkowski_dot(start_vec, end_vec)
#         if prod > -1.0: prod = -1.0 - 1e-7
#         total_dist = np.arccosh(-prod)
        
#         if total_dist < stop_distance_buffer: continue # 距离太近不画

#         # 计算停止比例 (stop_ratio)
#         current_stop_ratio = stop_distance_buffer / total_dist
        
#         # 2. 获取测地线路径点 (Curve)
#         gx, gy, gz = get_geodesic_path(start_vec, end_vec, num_points=60, stop_ratio=current_stop_ratio)
        
#         # 3. 绘制弯曲的虚线
#         ax.plot(gx, gy, gz, 
#                 color=pure_black, 
#                 linestyle=(0, (3, 2)), # 虚线样式
#                 linewidth=1.2,         # 粗细
#                 alpha=0.9, 
#                 zorder=300)
        
#         # 4. 计算箭头方向 (Tangent)
#         # 既然是曲线，箭头的方向应该是曲线末端的切线方向
#         # 我们取路径最后两个点来近似切向量
#         if len(gx) >= 2:
#             dx = gx[-1] - gx[-2]
#             dy = gy[-1] - gy[-2]
#             dz = gz[-1] - gz[-2]
            
#             # 归一化切向量
#             norm = np.sqrt(dx**2 + dy**2 + dz**2)
#             if norm > 0:
#                 ux, uy, uz = dx/norm, dy/norm, dz/norm
#             else:
#                 ux, uy, uz = 0, 0, 0

#             # 5. 绘制箭头
#             ax.quiver(gx[-1], gy[-1], gz[-1], ux, uy, uz, 
#                       length=arrow_len, normalize=True,
#                       color=pure_black, 
#                       edgecolor=pure_black,
#                       facecolor=pure_black,
#                       arrow_length_ratio=0.4, 
#                       linewidth=0.8, 
#                       alpha=1.0, 
#                       zorder=301)

# def draw_pairwise_geodesics(ax, xs, ys, zs, color='gray', linewidth=0.8, alpha=0.5):
#     """
#     绘制特征点两两之间的测地线连接
#     """
#     points = list(zip(xs, ys, zs))
#     num_points = len(points)
    
#     for i in range(num_points):
#         for j in range(i + 1, num_points):
#             p1 = np.array(points[i])
#             p2 = np.array(points[j])
            
#             # 获取路径，不需要提前停止
#             gx, gy, gz = get_geodesic_path(p1, p2, num_points=30, stop_ratio=0.0)
            
#             ax.plot(gx, gy, gz, 
#                     color=color, 
#                     linestyle=(0, (1, 3)), # 细实线表示内部关系
#                     linewidth=linewidth, 
#                     alpha=alpha, 
#                     zorder=90) # 层级略低于点(100)

# def plot_hyperbolic_final_with_arrows():
#     # 保存路径
#     script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
#     save_dir = os.path.join(script_dir, "hyperboloid_imgs")
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)

#     # ================= 1. 场景参数 =================
#     view_elev = 25   
#     view_azim = 120   
#     xy_limit = 3.0   
#     grid_density = 4 
#     line_width = 0.05 
#     grid_edge_color = (0, 0, 0, 0.15) 
#     surface_opacity = 0.4 
#     point_size = 150 
#     proto_size = 400 

#     colors = ["#151515", "#FAFAFA"]    
#     cmap_name = "hyperbolic_fade"
#     custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

#     # ================= 2. 绘图初始化 =================
#     fig = plt.figure(figsize=(10, 8), dpi=300)
#     fig.patch.set_alpha(0.0)
#     ax = fig.add_subplot(111, projection='3d')
#     ax.patch.set_alpha(0.0)
#     ax.zaxis.set_rotate_label(False)
#     ax.xaxis.set_rotate_label(False)
#     ax.yaxis.set_rotate_label(False)
#     ax.set_axis_off()

#     # ================= 3. 绘制曲面 =================
#     x = np.linspace(-xy_limit, xy_limit, 120)
#     y = np.linspace(-xy_limit, xy_limit, 120)
#     X, Y = np.meshgrid(x, y)
#     Z = np.sqrt(1 + X**2 + Y**2)

#     ax.plot_surface(X, Y, Z, 
#                     cmap=custom_cmap,
#                     edgecolor=grid_edge_color,
#                     alpha=surface_opacity,
#                     rstride=grid_density,  
#                     cstride=grid_density,
#                     linewidth=line_width,
#                     antialiased=True,
#                     shade=False,
#                     zorder=1)

#     # ================= 4. 生成数据点 =================
#     # --- 类别 A（黄色，中心）---
#     cx_a, cy_a = 0.0, 0.0
#     feat_x_a, feat_y_a, feat_z_a = generate_cluster(cx_a, cy_a, num_points=4, spread=0.2, random_seed=2)
#     proto_z_a = np.sqrt(1 + cx_a**2 + cy_a**2) + 0.0 

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

#     # ================= 5. 绘制点 =================
#     ax.scatter3D(feat_x_a, feat_y_a, feat_z_a, s=point_size, c='#CCE4FF', marker='o', 
#                edgecolors='#000000', linewidth=1.2, zorder=100, label='Class A')
#     # ax.scatter3D(feat_x_b_1, feat_y_b_1, feat_z_b_1, s=point_size, c='#E53935', marker='o', 
#     #            edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
#     # ax.scatter3D(feat_x_b_2, feat_y_b_2, feat_z_b_2, s=point_size, c='#E53935', marker='o', 
#     #            edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
#     # ax.scatter3D(feat_x_c, feat_y_c, feat_z_c, s=point_size, c='#1E88E5', marker='o', 
#     #            edgecolors='#000000', linewidth=1.2, zorder=100, label='Class C')
#     # ax.scatter3D(feat_x_d, feat_y_d, feat_z_d, s=point_size, c='#43A047', marker='o', 
#     #            edgecolors='#000000', linewidth=1.2, zorder=100, label='Class D')

#     # # 原型点
#     ax.scatter3D([cx_a], [cy_a], [proto_z_a], s=proto_size, c='#FFC107', marker='*', 
#                edgecolors='#000000', linewidth=1.0, zorder=200, label='Proto A')

#     # ================= 6. 绘制测地线连接 (核心修改) =================
#     link_color = '#000000'
    
#     # 绘制特征点两两之间的测地线 (Pairwise Distance)
#     # 用细线，表示“关系计算”
#     # draw_pairwise_geodesics(ax, feat_x_a, feat_y_a, feat_z_a, color='#000000', linewidth=1.5, alpha=0.5)

#     # 绘制 A 类指向原型的连接线 (测地线)
#     draw_geodesic_arrows(ax, feat_x_a, feat_y_a, feat_z_a, cx_a, cy_a, proto_z_a, color=link_color)
    
#     # 绘制其他类指向原型的线 (如果需要)
#     # draw_geodesic_arrows(ax, feat_x_b_1, feat_y_b_1, feat_z_b_1, cx_a, cy_a, proto_z_a, color=link_color)
#     # draw_geodesic_arrows(ax, feat_x_b_2, feat_y_b_2, feat_z_b_2, cx_a, cy_a, proto_z_a, color=link_color)
#     # draw_geodesic_arrows(ax, feat_x_c, feat_y_c, feat_z_c, cx_a, cy_a, proto_z_a, color=link_color)
#     # draw_geodesic_arrows(ax, feat_x_d, feat_y_d, feat_z_d, cx_a, cy_a, proto_z_a, color=link_color)

#     # ================= 7. 收尾 =================
#     ax.view_init(elev=view_elev, azim=view_azim)
#     ax.set_zlim(1, 3.5)
#     ax.set_box_aspect((1, 1, 0.6))
    
#     # 去除多余边距
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, "hyperbolic_pairwise.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
#     plt.savefig(os.path.join(save_dir, "hyperbolic_pairwise.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
#     print("生成完毕：已绘制精确的洛伦兹测地线连接")
#     plt.show()

# if __name__ == "__main__":
#     plot_hyperbolic_final_with_arrows()


import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

# ================= 1. 自定义3D箭头类（核心） =================
class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        # 强制箭头填充色/边缘色统一，避免空心/描边断裂
        target_color = kwargs.pop('color', 'black')
        if 'facecolor' not in kwargs:
            kwargs['facecolor'] = target_color
        if 'edgecolor' not in kwargs:
            kwargs['edgecolor'] = target_color
        kwargs['fill'] = True  # 实心箭头
        # 统一箭头线宽，与直线匹配
        if 'lw' not in kwargs and 'linewidth' not in kwargs:
            kwargs['lw'] = 1.0  # 与直线线宽一致
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        # 3D坐标转2D投影，确保箭头在3D空间正确显示
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

# ================= 工具函数：生成碗内点 =================
def generate_cluster(center_x, center_y, num_points=4, spread=0.15, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    x = np.random.normal(center_x, spread * 4.0, num_points)
    y = np.random.normal(center_y, spread * 2.0, num_points)
    # 保持较大的Z轴偏移，确保点悬浮在碗内
    z = np.sqrt(1 + x**2 + y**2) + 0.8 
    return x, y, z

# ================= 绘制防重叠连接线 (直线+Arrow3D箭头) =================
def draw_geodesic_arrows(ax, start_xs, start_ys, start_zs, end_x, end_y, end_z, color='black'):
    """
    绘制从特征点(start)指向原型(end)的直线+Arrow3D箭头（防重叠）
    核心修改：虚线→实线，线宽与箭头统一
    """
    stop_distance = 0.25    # 距离终点停止的缓冲区（防重叠）
    arrow_segment_len = 0.1 # Arrow3D箭头的线段长度（控制箭头位置）
    
    # 强制纯黑，确保样式统一
    pure_black = '#000000'

    for sx, sy, sz in zip(start_xs, start_ys, start_zs):
        # 1. 计算方向向量 (Start -> End)
        dx, dy, dz = end_x - sx, end_y - sy, end_z - sz
        full_dist = np.sqrt(dx**2 + dy**2 + dz**2)
        
        if full_dist < stop_distance: continue # 距离太近跳过

        # 2. 归一化方向向量
        ux, uy, uz = dx/full_dist, dy/full_dist, dz/full_dist
        
        # 3. 计算直线终点（提前停止）
        draw_dist = full_dist - stop_distance
        line_end_x = sx + ux * draw_dist
        line_end_y = sy + uy * draw_dist
        line_end_z = sz + uz * draw_dist
        
        # 4. 绘制直线（核心修改：虚线→实线）
        ax.plot([sx, line_end_x], [sy, line_end_y], [sz, line_end_z], 
                color=pure_black, 
                linestyle='-',  # 关键：从虚线改为实线
                linewidth=1.0,  # 直线宽度，与Arrow3D线宽统一
                alpha=1.0, 
                zorder=300)
        
        # 5. 计算Arrow3D箭头的起点/终点（在直线终点后延伸一小段）
        arrow_start_x = line_end_x
        arrow_start_y = line_end_y
        arrow_start_z = line_end_z
        
        arrow_end_x = line_end_x + ux * arrow_segment_len
        arrow_end_y = line_end_y + uy * arrow_segment_len
        arrow_end_z = line_end_z + uz * arrow_segment_len
        
        # 6. 创建并添加Arrow3D箭头
        arrow = Arrow3D(
            [arrow_start_x, arrow_end_x],  # x轴坐标
            [arrow_start_y, arrow_end_y],  # y轴坐标
            [arrow_start_z, arrow_end_z],  # z轴坐标
            mutation_scale=8,              # 箭头头部大小（适配图表）
            arrowstyle="-|>",              # 标准PPT风格箭头
            color=pure_black,              # 箭头颜色与直线统一
            alpha=1.0,
            zorder=301                     # 层级高于直线，避免遮挡
        )
        ax.add_artist(arrow)

def plot_hyperbolic_final_with_arrows():
    # 保存路径
    script_dir = os.path.dirname(os.path.abspath(__file__)) if __name__ == "__main__" else os.getcwd()
    save_dir = os.path.join(script_dir, "imgs")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ================= 1. 场景参数 =================
    view_elev = 25   
    view_azim = 120   
    xy_limit = 3.0   
    grid_density = 4 
    line_width = 0.05 
    grid_edge_color = (0, 0, 0, 0.15) 
    surface_opacity = 0.4 
    point_size = 150 
    proto_size = 400 

    # 论文标准渐变色
    colors = ["#151515", "#FAFAFA"]    
    cmap_name = "hyperbolic_fade"
    custom_cmap = mcolors.LinearSegmentedColormap.from_list(cmap_name, colors)

    # ================= 2. 绘图初始化 =================
    fig = plt.figure(figsize=(10, 8), dpi=300)
    fig.patch.set_alpha(0.0)  # 画布透明
    ax = fig.add_subplot(111, projection='3d')
    ax.patch.set_alpha(0.0)   # 坐标轴背景透明
    
    ax.zaxis.set_rotate_label(False)
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_rotate_label(False)
    ax.set_axis_off()

    # ================= 3. 绘制曲面 =================
    x = np.linspace(-xy_limit, xy_limit, 120)
    y = np.linspace(-xy_limit, xy_limit, 120)
    X, Y = np.meshgrid(x, y)
    Z = np.sqrt(1 + X**2 + Y**2)

    ax.plot_surface(X, Y, Z, 
                    cmap=custom_cmap,
                    edgecolor=grid_edge_color,
                    alpha=surface_opacity,
                    rstride=grid_density,  
                    cstride=grid_density,
                    linewidth=line_width,
                    antialiased=True,
                    shade=False,
                    zorder=1)

    # ================= 4. 生成数据点 =================
    # --- 类别 A（黄色，中心）---
    cx_a, cy_a = 0.0, 0.0
    feat_x_a, feat_y_a, feat_z_a = generate_cluster(cx_a, cy_a, num_points=4, spread=0.2, random_seed=2)
    proto_z_a = np.sqrt(1 + cx_a**2 + cy_a**2) + 0.0 # 原型贴地一点

    # --- 类别 B（红色，右侧）---
    cx_b_1, cy_b_1 = -0.2, 0.7
    feat_x_b_1, feat_y_b_1, feat_z_b_1 = generate_cluster(cx_b_1, cy_b_1, num_points=1, spread=0.15, random_seed=1)
    cx_b_2, cy_b_2 = -1.9, -1.5
    feat_x_b_2, feat_y_b_2, feat_z_b_2 = generate_cluster(cx_b_2, cy_b_2, num_points=1, spread=0.15, random_seed=1)
    
    # --- 类别 C（蓝色，左侧）---
    cx_c, cy_c = -1.0, 0.5
    feat_x_c, feat_y_c, feat_z_c = generate_cluster(cx_c, cy_c, num_points=1, spread=0.15, random_seed=1)

    # --- 类别 D（绿色，左侧）---
    cx_d, cy_d = 0.5, 2.0
    feat_x_d, feat_y_d, feat_z_d = generate_cluster(cx_d, cy_d, num_points=1, spread=0.10, random_seed=1)

    # ================= 5. 绘制点 =================
    # 特征点
    ax.scatter3D(feat_x_a, feat_y_a, feat_z_a, s=point_size, c='#FFC107', marker='o', 
               edgecolors='#000000', linewidth=1.2, zorder=100, label='Class A')
    ax.scatter3D(feat_x_b_1, feat_y_b_1, feat_z_b_1, s=point_size, c='#E53935', marker='o', 
               edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
    ax.scatter3D(feat_x_b_2, feat_y_b_2, feat_z_b_2, s=point_size, c='#E53935', marker='o', 
               edgecolors='#000000', linewidth=1.2, zorder=100, label='Class B')
    ax.scatter3D(feat_x_c, feat_y_c, feat_z_c, s=point_size, c='#1E88E5', marker='o', 
               edgecolors='#000000', linewidth=1.2, zorder=100, label='Class C')
    ax.scatter3D(feat_x_d, feat_y_d, feat_z_d, s=point_size, c='#43A047', marker='o', 
               edgecolors='#000000', linewidth=1.2, zorder=100, label='Class D')

    # 原型点
    ax.scatter3D([cx_a], [cy_a], [proto_z_a], s=proto_size, c='#FFC107', marker='*', 
               edgecolors='#000000', linewidth=1.0, zorder=200, label='Proto A')

    # ================= 6. 绘制连接线 (直线+Arrow3D箭头) =================
    link_color = '#000000'
    # 绘制 A 类连接线
    draw_geodesic_arrows(ax, feat_x_a, feat_y_a, feat_z_a, cx_a, cy_a, proto_z_a, color=link_color)
    # 绘制其他类别指向中心的连接线
    draw_geodesic_arrows(ax, feat_x_b_1, feat_y_b_1, feat_z_b_1, cx_a, cy_a, proto_z_a, color=link_color)
    draw_geodesic_arrows(ax, feat_x_b_2, feat_y_b_2, feat_z_b_2, cx_a, cy_a, proto_z_a, color=link_color)
    draw_geodesic_arrows(ax, feat_x_c, feat_y_c, feat_z_c, cx_a, cy_a, proto_z_a, color=link_color)
    draw_geodesic_arrows(ax, feat_x_d, feat_y_d, feat_z_d, cx_a, cy_a, proto_z_a, color=link_color)

    # ================= 7. 收尾 =================
    ax.view_init(elev=view_elev, azim=view_azim)
    ax.set_zlim(1, 3.5)
    ax.set_box_aspect((1, 1, 0.6))
    ax.draw(renderer=fig.canvas.get_renderer())

    print("生成完毕：已替换为直线+Arrow3D箭头（防重叠）")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "hyperbolic_final_line_arrow3d.svg"), transparent=True, format="svg", bbox_inches='tight', pad_inches=0)
    plt.savefig(os.path.join(save_dir, "hyperbolic_final_line_arrow3d.png"), transparent=True, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.show()

if __name__ == "__main__":
    plot_hyperbolic_final_with_arrows()