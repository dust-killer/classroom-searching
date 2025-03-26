import cv2
import tkinter as tk
from PIL import Image, ImageTk
import os
import heapq
import time

class Node:
    def __init__(self, z, x, y, g=float('inf'), h=float('inf'), parent=None):
        self.z = z
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

    def __lt__(self, other):
        return self.f < other.f


# 池化大小
POOL_SIZE = 20


# 池化函数
def pool_map(map_data):
    """
    对地图数据进行池化处理

    :param map_data: 输入的地图数据，是一个二维列表
    :return: 池化后的地图数据，也是一个二维列表
    """
    rows = len(map_data)
    cols = len(map_data[0])
    pooled_map = []
    for i in range(0, rows, POOL_SIZE):
        row = []
        for j in range(0, cols, POOL_SIZE):
            sub_map = [map_data[ii][jj] for ii in range(i, min(i + POOL_SIZE, rows)) for jj in
                       range(j, min(j + POOL_SIZE, cols))]
            if any(cell == 2 for cell in sub_map):
                row.append(2)
            else:
                row.append(0)
        pooled_map.append(row)
    return pooled_map


# 读取图片并转换为地图
def image_to_map(image_path, index):
    """
    读取图片并将其转换为地图表示

    :param image_path: 图片文件的路径
    :param index: 图片的索引，用于保存中间处理结果时的命名
    :return: 转换后的地图数据（二维列表），如果读取失败则返回 None
    """
    print(f"尝试读取图片: {image_path}")
    if not os.path.exists(image_path):
        print(f"文件不存在: {image_path}")
        return None
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    if img is None:
        print(f"无法读取图片: {image_path}，可能是格式问题或路径错误")
        return None
    _, binary_img = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
    rows, cols = binary_img.shape
    building_map = []
    for i in range(rows):
        row = []
        for j in range(cols):
            if binary_img[i, j] == 255:
                row.append(0)  # 白色表示可通行区域
            else:
                row.append(2)  # 黑色表示墙壁
        building_map.append(row)
    # 池化处理
    building_map = pool_map(building_map)
    print(f"成功读取并处理图片 {image_path}")
    return building_map

# 启发式函数
def heuristic(node, target):
    """
    计算节点到目标的启发式距离

    :param node: 当前节点对象
    :param target: 目标节点的坐标 (z, x, y)
    :return: 启发式距离值
    """
    return abs(node.x - target[1]) + abs(node.y - target[2])


# 寻找最近的楼梯
def find_nearest_stair(current, stairs):
    """
    在当前楼层的楼梯列表中找到距离当前点最近的楼梯

    :param current: 当前点的坐标 (z, x, y)
    :param stairs: 当前楼层的楼梯列表，每个元素是一个包含楼梯信息的元组
    :return: 最近楼梯的坐标 (z, x, y)，如果没有楼梯则返回 None
    """
    if not stairs:
        print("当前楼层没有楼梯信息")
        return None
    min_distance = float('inf')
    nearest_stair = None
    for stair in stairs:
        distance = heuristic(Node(current[0], current[1], current[2]), (stair[0], stair[1], stair[2]))
        if distance < min_distance:
            min_distance = distance
            nearest_stair =(stair[0], stair[1], stair[2])
    return nearest_stair


# 平滑化的 A* 算法
# 平滑化的 A* 算法
# 平滑化的 A* 算法
# 平滑化的 A* 算法
def smoothed_a_star(start, target, building_map, stairs):
    """
    实现平滑化的 A* 寻路算法

    :param start: 起始点的坐标 (z, x, y)
    :param target: 目标点的坐标 (z, x, y)
    :param building_map: 地图数据，是一个三维列表，表示各楼层的地图
    :param stairs: 楼梯信息字典，键为楼层号，值为该楼层楼梯信息的列表
    :return: 从起始点到目标点的路径（坐标列表），如果没有找到路径则返回 None
    """
    num_floors = len(building_map)
    rows = len(building_map[start[0]])
    cols = len(building_map[start[0]][0])

    open_list = []
    closed_set = set()
    searched_nodes = {i: [] for i in range(num_floors)}

    start_node = Node(start[0], start[1], start[2], g=0, h=heuristic(Node(start[0], start[1], start[2]), target))
    heapq.heappush(open_list, start_node)

    last_print_time = time.time()
    tim = 0
    start_time = time.time()  # 记录寻路开始时间

    total_path = []

    current_start = start
    while current_start[0] != target[0]:
        nearest_stair = find_nearest_stair(current_start, stairs[current_start[0]])
        if nearest_stair is None:
            print(f"当前楼层 {current_start[0]} 没有找到合适的楼梯")
            return None, searched_nodes

        # 临时的 open_list 和 closed_set 用于当前楼层的寻路
        temp_open_list = []
        temp_closed_set = set()
        temp_start_node = Node(current_start[0], current_start[1], current_start[2], g=0,
                               h=heuristic(Node(current_start[0], current_start[1], current_start[2]), nearest_stair))
        heapq.heappush(temp_open_list, temp_start_node)

        while temp_open_list:
            current_time = time.time()
            if current_time - last_print_time >= 5:
                tim += 5
                current = temp_open_list[0]
                print("寻路仍在进行中...已花费" + str(tim) + "秒\n剩余节点数" + str(len(temp_open_list)) + "\n已寻找" + str(
                    len(temp_closed_set)) + "\n估计距离" + str(current.f))
                last_print_time = current_time

            if current_time - start_time > 60:  # 判断是否超时
                print("寻路超时，停止寻路。")
                return None, searched_nodes

            current = heapq.heappop(temp_open_list)

            if (current.z, current.x, current.y) == nearest_stair:
                path = []
                while current:
                    path.append((current.z, current.x, current.y))
                    current = current.parent
                total_path.extend(path[::-1][:-1])  # 不添加楼梯节点，因为后续会单独处理
                break

            temp_closed_set.add((current.z, current.x, current.y))
            searched_nodes[current.z].append((current.x, current.y))  # 记录已搜索节点

            # 实时更新显示
            show_path_on_floor(current.z, searched_nodes[current.z], [])

            horizontal_directions = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
            all_directions = horizontal_directions

            for dz, dx, dy in all_directions:
                new_z, new_x, new_y = current.z + dz, current.x + dx, current.y + dy
                if (
                        0 <= new_z < num_floors
                        and 0 <= new_x < rows
                        and 0 <= new_y < cols
                        and (new_z, new_x, new_y) not in temp_closed_set
                ):
                    if building_map[new_z][new_x][new_y] != 2:
                        tentative_g = current.g + 1
                        neighbor = Node(new_z, new_x, new_y)
                        if tentative_g < neighbor.g:
                            neighbor.parent = current
                            neighbor.g = tentative_g
                            neighbor.h = heuristic(neighbor, nearest_stair)
                            neighbor.f = neighbor.g + neighbor.h
                            heapq.heappush(temp_open_list, neighbor)
        else:
            print(f"未找到从当前点到楼梯的路径，当前点: {current_start}")
            return None, searched_nodes

        # 找到下一层对应的楼梯
        next_floor = current_start[0] + 1 if current_start[0] < target[0] else current_start[0] - 1
        found_next_stair = False
        for stair_info in stairs[next_floor]:
            if stair_info[0] == next_floor and stair_info[4] == nearest_stair[1] and stair_info[5] == nearest_stair[2]:
                current_start = stair_info[:3]
                total_path.append(current_start)
                found_next_stair = True
                break
        if not found_next_stair:
            print(f"在楼层 {next_floor} 未找到对应的楼梯")
            return None, searched_nodes

    # 在目标楼层进行寻路
    open_list = []
    closed_set = set()
    start_node = Node(current_start[0], current_start[1], current_start[2], g=0,
                      h=heuristic(Node(current_start[0], current_start[1], current_start[2]), target))
    heapq.heappush(open_list, start_node)

    while open_list:
        current_time = time.time()
        if current_time - last_print_time >= 5:
            tim += 5
            current = open_list[0]
            print("寻路仍在进行中...已花费" + str(tim) + "秒\n剩余节点数" + str(len(open_list)) + "\n已寻找" + str(
                len(closed_set)) + "\n估计距离" + str(current.f))
            last_print_time = current_time

        if current_time - start_time > 60:  # 判断是否超时
            print("寻路超时，停止寻路。")
            return None, searched_nodes

        current = heapq.heappop(open_list)

        if (current.z, current.x, current.y) == target:
            path = []
            while current:
                path.append((current.z, current.x, current.y))
                current = current.parent
            total_path.extend(path[::-1])
            return total_path, searched_nodes

        closed_set.add((current.z, current.x, current.y))
        searched_nodes[current.z].append((current.x, current.y))  # 记录已搜索节点

        # 实时更新显示
        show_path_on_floor(current.z, searched_nodes[current.z], [])

        horizontal_directions = [(0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
        all_directions = horizontal_directions

        for dz, dx, dy in all_directions:
            new_z, new_x, new_y = current.z + dz, current.x + dx, current.y + dy
            if (
                    0 <= new_z < num_floors
                    and 0 <= new_x < rows
                    and 0 <= new_y < cols
                    and (new_z, new_x, new_y) not in closed_set
            ):
                if building_map[new_z][new_x][new_y] != 2:
                    tentative_g = current.g + 1
                    neighbor = Node(new_z, new_x, new_y)
                    if tentative_g < neighbor.g:
                        neighbor.parent = current
                        neighbor.g = tentative_g
                        neighbor.h = heuristic(neighbor, target)
                        neighbor.f = neighbor.g + neighbor.h
                        heapq.heappush(open_list, neighbor)

    return None, searched_nodes  # 未找到路径时也返回已搜索节点


# 显示指定楼层的路径

# 鼠标点击事件处理
def on_click(event):
    global start_point
    try:
        x, y = event.x, event.y
        floor = int(floor_var.get())
        # 获取原始图片尺寸
        original_img = cv2.imread(image_paths[floor], cv2.IMREAD_GRAYSCALE)
        original_height, original_width = original_img.shape

        # 获取缩放后图片尺寸
        img = cv2.imread(image_paths[floor], cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
        img = img.resize((500, 500), Image.LANCZOS)
        scaled_width, scaled_height = img.size

        # 计算缩放比例
        width_scale = original_width / scaled_width
        height_scale = original_height / scaled_height

        # 调整鼠标点击坐标
        original_x = int(x * width_scale)
        original_y = int(y * height_scale)

        # 调整到池化后的坐标
        original_x = original_x // POOL_SIZE
        original_y = original_y // POOL_SIZE

        start_point = (floor, original_y, original_x)
        start_point1 = (floor, original_y * POOL_SIZE, original_x * POOL_SIZE)
        status_label.config(text=f"起始位置已选择: {start_point1}")
        print(f"鼠标点击位置: ({x}, {y}), 楼层: {floor}")
        print(f"调整后位置: ({original_x}, {original_y}), 楼层: {floor}")
        map_label.pack_forget()
        back_button.pack_forget()
        find_button.pack()
    except Exception as e:
        print(f"鼠标点击事件处理出错: {e}")

def show_path_on_floor(floor, current_searched_nodes, current_path):
    global map_label
    img = cv2.imread(image_paths[floor], cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取图像: {image_paths[floor]}")
        return

    print(f"绘制楼层 {floor} 的已搜索节点: {current_searched_nodes}")
    print(f"绘制楼层 {floor} 的路径: {current_path}")

    height, width, _ = img.shape

    # 绘制已搜索的节点
    for x, y in current_searched_nodes:
        x_original = x * POOL_SIZE + POOL_SIZE // 2
        y_original = y * POOL_SIZE + POOL_SIZE // 2
        if 0 <= x_original < height and 0 <= y_original < width:
            cv2.circle(img, (y_original, x_original), 2, (0, 255, 0), -1)  # 绿色圆圈表示已搜索节点
        else:
            print(f"已搜索节点坐标 ({x_original}, {y_original}) 超出图像范围")

    prev_point = None
    for index, (_, x, y) in enumerate(current_path):
        x_original = x * POOL_SIZE + POOL_SIZE // 2
        y_original = y * POOL_SIZE + POOL_SIZE // 2
        current_point = (y_original, x_original)
        if 0 <= x_original < height and 0 <= y_original < width:
            if prev_point is not None:
                # 使用 cv2.arrowedLine 绘制箭头
                print(f"从 {prev_point} 绘制箭头到 {current_point}")
                cv2.arrowedLine(img, prev_point, current_point, (255, 0, 0), 1, tipLength=0.5)
            prev_point = current_point
        else:
            print(f"路径点坐标 ({x_original}, {y_original}) 超出图像范围")

    # 检查 cv2.imwrite 是否成功
    if cv2.imwrite("path_image.jpg", img):
        img = cv2.imread("path_image.jpg", cv2.IMREAD_COLOR)
        if img is not None:
            img = Image.fromarray(img)
            img = img.resize((500, 500), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            map_label.config(image=img)
            map_label.image = img
            if back_button.winfo_ismapped() == 0:
                back_button.pack()
            map_label.pack()
            root.update()  # 强制更新界面
        else:
            print("无法读取保存的图像文件: path_image.jpg")
    else:
        print("无法保存图像文件: path_image.jpg")
# 寻路按钮点击事件处理
def find_path():
    global start_point, map_label, current_floor, path_per_floor, searched_nodes
    if start_point is None:
        status_label.config(text="请先选择起始位置")
        return
    try:
        target_floor = int(target_floor_entry.get())
        target_x = int(target_x_entry.get())
        target_y = int(target_y_entry.get())

        target_point = (target_floor, target_x, target_y)

        # 检查起始点和目标点是否在可通行区域内
        if building_map[start_point[0]][start_point[1]][start_point[2]] != 0:
            status_label.config(text="起始点不在可通行区域内")
            return
        if building_map[target_point[0]][target_point[1]][target_point[2]] != 0:
            status_label.config(text="目标点不在可通行区域内")
            return

        result = smoothed_a_star(start_point, target_point, building_map, stairs)
        if result is None:
            status_label.config(text="未找到路径，请检查地图或起始/目标位置。")
            return
        path, searched_nodes = result
        path_per_floor = {i: [] for i in range(len(image_paths))}  # 初始化 path_per_floor 字典
        if path:
            unique_path = []
            for p in path:
                if isinstance(p, list):
                    for sub_p in p:
                        floor = sub_p[0]
                        if sub_p not in unique_path:
                            unique_path.append(sub_p)
                            if floor not in path_per_floor:
                                path_per_floor[floor] = []
                            path_per_floor[floor].append(sub_p)
                else:
                    floor = p[0]
                    if p not in unique_path:
                        unique_path.append(p)
                        if floor not in path_per_floor:
                            path_per_floor[floor] = []
                        path_per_floor[floor].append(p)

            current_floor = start_point[0]
            show_path_on_floor(current_floor, searched_nodes[current_floor], path_per_floor[current_floor])
            if up_button.winfo_ismapped() == 0:
                up_button.pack()
            if down_button.winfo_ismapped() == 0:
                down_button.pack()
            status_label.config(text="找到路径！")
            root.update()
        else:
            status_label.config(text="未找到路径。")
    except ValueError:
        status_label.config(text="请输入有效的目标位置")

# 寻路按钮点击事件处理
def find_path():
    global start_point, map_label, current_floor, path_per_floor, searched_nodes
    if start_point is None:
        status_label.config(text="请先选择起始位置")
        return
    try:
        target_floor = int(target_floor_entry.get())
        target_x = int(target_x_entry.get())
        target_y = int(target_y_entry.get())

        target_point = (target_floor, target_x, target_y)

        # 检查起始点和目标点是否在可通行区域内
        if building_map[start_point[0]][start_point[1]][start_point[2]] != 0:
            status_label.config(text="起始点不在可通行区域内")
            return
        if building_map[target_point[0]][target_point[1]][target_point[2]] != 0:
            status_label.config(text="目标点不在可通行区域内")
            return

        result = smoothed_a_star(start_point, target_point, building_map, stairs)
        if result is None:
            status_label.config(text="未找到路径，请检查地图或起始/目标位置。")
            return
        path, searched_nodes = result
        path_per_floor = {i: [] for i in range(len(image_paths))}  # 初始化 path_per_floor 字典
        if path:
            for p in path:
                floor = p[0]
                if floor not in path_per_floor:
                    path_per_floor[floor] = []
                path_per_floor[floor].append(p)
            current_floor = start_point[0]
            show_path_on_floor(current_floor, searched_nodes[current_floor], path_per_floor[current_floor])
            if up_button.winfo_ismapped() == 0:
                up_button.pack()
            if down_button.winfo_ismapped() == 0:
                down_button.pack()
            status_label.config(text="找到路径！")
            root.update()
        else:
            status_label.config(text="未找到路径。")
    except ValueError:
        status_label.config(text="请输入有效的目标位置")
# 选择楼层事件处理
def select_floor(*args):
    if back_button.winfo_ismapped() == 0:
        back_button.pack()
    floor = int(floor_var.get())
    img = cv2.imread(image_paths[floor], cv2.IMREAD_GRAYSCALE)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = Image.fromarray(img)
    img = img.resize((500, 500), Image.LANCZOS)
    img = ImageTk.PhotoImage(img)
    map_label.config(image=img)
    map_label.image = img
    map_label.pack()
    find_button.pack_forget()
    floor_menu.update()  # 强制更新 OptionMenu 显示


# 返回按钮点击事件处理
def hide_map():
    map_label.pack_forget()
    up_button.pack_forget()
    down_button.pack_forget()
    if back_button.winfo_ismapped():
        back_button.pack_forget()


# 教室确认按钮点击事件处理
def confirm_classroom():
    classroom = classroom_entry.get()
    if classroom in classroom_coordinates:
        target_floor, target_x, target_y = classroom_coordinates[classroom]
        target_floor_entry.delete(0, tk.END)
        target_floor_entry.insert(0, target_floor)
        target_x_entry.delete(0, tk.END)
        target_x_entry.insert(0, target_x)
        target_y_entry.delete(0, tk.END)
        target_y_entry.insert(0, target_y)
        status_label.config(text=f"目标位置已设置为 {classroom} 的坐标")
    else:
        status_label.config(text=f"未找到教室 {classroom} 的坐标")


# 上一层按钮点击事件处理
# 上一层按钮点击事件处理
# 上一层按钮点击事件处理
def go_up():
    global current_floor
    if current_floor < len(image_paths) - 1:
        current_floor += 1
        if current_floor not in path_per_floor:
            path_per_floor[current_floor] = []
        if current_floor not in searched_nodes:
            searched_nodes[current_floor] = []
        show_path_on_floor(current_floor, searched_nodes[current_floor], path_per_floor[current_floor])


# 下一层按钮点击事件处理
def go_down():
    global current_floor
    if current_floor > 0:
        current_floor -= 1
        if current_floor not in path_per_floor:
            path_per_floor[current_floor] = []
        if current_floor not in searched_nodes:
            searched_nodes[current_floor] = []
        show_path_on_floor(current_floor, searched_nodes[current_floor], path_per_floor[current_floor])
# 直接指定教学楼地图图片路径
image_paths = [
    r"0.jpg",
    r"1.jpg",
    r"2.jpg",
    r"3.jpg",
    r"4.jpg",
    r"5.jpg"
    # 可以根据实际情况添加更多楼层的图片路径
]

print(list(range(len(image_paths))))  # 打印楼层范围，检查是否正确

root = tk.Tk()
root.title("教学楼寻路软件")

if not image_paths:
    root.destroy()
    raise ValueError("未指定任何图片路径")

building_map = []
for i in range(len(image_paths)):
    map_data = image_to_map(image_paths[i], i)
    if map_data is not None:
        building_map.append(map_data)

if not building_map:
    root.destroy()
    raise ValueError("所有图片均无法读取，请检查图片路径和格式。")

# 创建教室坐标字典
classroom_coordinates = {
    "6A015": (0, 16, 54),
    "6A014": (0, 20, 54),
    "6A013": (0, 26, 54),
    "6A012": (0, 30, 53),
    "6A011": (0, 36, 54),
    "6A010": (0, 40, 53),
    "6A009": (0, 44, 52),
    "6A008": (0, 44, 52),
    "6A007": (0, 50, 51),
    "6A006": (0, 50, 46),
    "6A005": (0, 52, 41),
    "6A004": (0, 53, 36),
    "6A003": (0, 53, 31),
    "6A002": (0, 54, 26),
    "6A001": (0, 55, 21),
    "6A016": (0, 25, 50),
    "6A017": (0, 46, 47),
    "6A018": (0, 50, 27),
    "6A101": (1, 47, 20),
    "6A102": (1, 47, 25),
    "6A103": (1, 47, 30),
    "6A104": (1, 47, 35),
    "6A105": (1, 47, 40),
    "6A106": (1, 47, 42),
    "6A108": (1, 44, 49),
    "6A109": (1, 40, 50),
    "6A110": (1, 40, 50),
    "6A111": (1, 36, 50),
    "6A112": (1, 32, 50),
    "6A113": (1, 26, 50),
    "6A114": (1, 21, 50),
    "6A115": (1, 18, 50),
    "6A116": (1, 21, 43),
    "6A117": (1, 39, 42),
    "6A118": (1, 42, 26),
    "6C102": (1, 30, 17),
    "6C101": (1, 26, 17),
    "6B100": (1, 3, 38),
    "6B101": (1, 3, 38),
    "6B102": (1, 3, 38),
    "6B103": (1, 7, 33),
    "6B104": (1, 7, 28),
    "6B105": (1, 7, 23),
    "6B106": (1, 7, 18),
    "6B108": (1, 9, 19),
    "6B109": (1, 12, 22),
    "6B110": (1, 15, 25),
    "6B111": (1, 19, 23),
    "6B112": (1, 15, 27),
    "6B113": (1, 11, 30)
}

# 创建楼梯坐标字典，每个楼层对应多个楼梯及其下一层的对应位置
stairs = {
    0: [(0, 58, 10, 1, 58, 11), (0, 10, 59, 1, 10, 58),(0,53,48,1,47,45),(0,48,53,1,42,49)],
    1: [(1, 58, 11,0, 58, 10), ( 1, 10, 58,0, 10, 59),(1,47,45,0,53,48),(1,42,49,0,48,53)],
    2: [(2, 300, 300, 1, 300, 300), (2, 700, 700, 3, 700, 700)],
    3: [(3, 700, 700, 2, 700, 700), (3, 400, 400, 4, 400, 400)],
    4: [(4, 400, 400, 3, 400, 400), (4, 600, 600, 5, 600, 600)],
    5: [(5, 600, 600, 4, 600, 600)]
}

# 创建 GUI 组件
floor_var = tk.StringVar()
floor_var.set("0")
start_point = None
current_floor = 0
path_per_floor = {}

status_label = tk.Label(root, text="请选择起始位置和目标位置")
status_label.pack()

floor_label = tk.Label(root, text="选择楼层:")
floor_label.pack()
floor_menu = tk.OptionMenu(root, floor_var, *range(len(image_paths)), command=select_floor)
floor_menu.pack()

map_label = tk.Label(root)
map_label.bind("<Button-1>", on_click)

find_button = tk.Button(root, text="寻路", command=find_path)

# 创建返回按钮
back_button = tk.Button(root, text="返回", command=hide_map)

# 创建上一层和下一层按钮
up_button = tk.Button(root, text="上一层", command=go_up)
down_button = tk.Button(root, text="下一层", command=go_down)

target_frame = tk.Frame(root)
target_frame.pack()
target_floor_label = tk.Label(target_frame, text="目标楼层:")
target_floor_label.pack(side=tk.LEFT)
target_floor_entry = tk.Entry(target_frame)
target_floor_entry.pack(side=tk.LEFT)
target_x_label = tk.Label(target_frame, text="目标行:")
target_x_label.pack(side=tk.LEFT)
target_x_entry = tk.Entry(target_frame)
target_x_entry.pack(side=tk.LEFT)
target_y_label = tk.Label(target_frame, text="目标列:")
target_y_label.pack(side=tk.LEFT)
target_y_entry = tk.Entry(target_frame)
target_y_entry.pack(side=tk.LEFT)

# 创建教室输入框和确认按钮
classroom_frame = tk.Frame(root)
classroom_frame.pack()
classroom_label = tk.Label(classroom_frame, text="输入教室名称:")
classroom_label.pack(side=tk.LEFT)
classroom_entry = tk.Entry(classroom_frame)
classroom_entry.pack(side=tk.LEFT)
confirm_button = tk.Button(classroom_frame, text="确认", command=confirm_classroom)
confirm_button.pack(side=tk.LEFT)

searched_nodes = {i: [] for i in range(len(image_paths))}  # 全局变量用于存储已搜索节点

root.mainloop()
