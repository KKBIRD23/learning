import sys
import json
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem,
                            QTableWidget, QTableWidgetItem, QSplitter, QVBoxLayout,
                            QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton,
                            QCheckBox, QTabWidget, QToolBar, QAction, QGraphicsView,
                            QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
                            QFileDialog, QMenu, QStatusBar, QHeaderView, QDialog, QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QColor, QPen, QBrush, QPainter, QIcon
import uiautomation as auto
import time
from typing import Optional, List, Dict, Any

class ScannerThread(QThread):
    scan_finished = pyqtSignal(object)
    
    def __init__(self, scanner, title, class_name, fuzzy, regex):
        super().__init__()
        self.scanner = scanner
        self.title = title
        self.class_name = class_name
        self.fuzzy = fuzzy
        self.regex = regex
    
    def run(self):
        window = self.scanner.find_main_window(
            title=self.title,
            class_name=self.class_name,
            fuzzy=self.fuzzy,
            regex=self.regex)
        
        if window:
            structure = self.scanner.explore_control_structure(window, max_depth=20)
            self.scan_finished.emit(structure)

class GUIScanner:
    """原有的扫描器类，保持不变"""
    def __init__(self):
        self.scan_count = 0
        self.found_controls = []
        auto.uiautomation.SetGlobalSearchTimeout(10)

    def list_all_windows(self) -> List[Dict[str, str]]:
        """列出系统中所有窗口"""
        windows = []
        root = auto.uiautomation.GetRootControl()
        for ctrl, depth in auto.uiautomation.WalkControl(root, maxDepth=1):
            try:
                if self.is_valid_main_window(ctrl):
                    windows.append({
                        "name": ctrl.Name or "无标题窗口",
                        "class_name": ctrl.ClassName or "未知类名"
                    })
            except Exception:
                continue
        return windows

    def is_match(self, value: str, pattern: str, fuzzy: bool = False, regex: bool = False) -> bool:
        if not value:
            return False
        if regex:
            return re.search(pattern, value, re.IGNORECASE) is not None
        elif fuzzy:
            return pattern.lower() in value.lower()
        else:
            return value == pattern
        
    def find_main_window(self, title: str, class_name: str = "", fuzzy=False, regex=False):
        print(f"\n正在查找窗口: '{title}'...")
        try:
            if class_name:
                window = auto.WindowControl(searchDepth=1, Name=title, ClassName=class_name)
            else:
                window = auto.WindowControl(searchDepth=1, Name=title)
            if window.Exists(3, 1):
                print(f"\n✅ 找到目标窗口: {window.Name} (类名: {window.ClassName})")
                return window
        except Exception as e:
            print("直接查找窗口时发生异常：", e)

        candidates = []
        root = auto.uiautomation.GetRootControl()
        for ctrl, depth in auto.uiautomation.WalkControl(root, maxDepth=10):
            self.scan_count += 1
            try:
                if not self.is_valid_main_window(ctrl):
                    continue
                if class_name and not self.is_match(ctrl.ClassName, class_name, fuzzy, regex):
                    continue
                if self.is_match(ctrl.Name, title, fuzzy, regex):
                    candidates.append(ctrl)
            except Exception as e:
                continue

        if not candidates:
            print("❌ 未找到匹配的窗口。")
            return None

        if len(candidates) == 1:
            target = candidates[0]
        else:
            print("🔍 找到多个候选窗口，请选择：")
            for i, ctrl in enumerate(candidates):
                print(f"[{i}] 标题: {ctrl.Name}, 类名: {ctrl.ClassName}, 控件类型: {ctrl.ControlTypeName}")
            try:
                idx = int(input("请输入要选择的窗口序号 (默认0): ").strip() or 0)
                target = candidates[idx]
            except:
                print("⚠️ 输入无效，默认选择第一个窗口。")
                target = candidates[0]

        print(f"\n✅ 找到目标窗口: {target.Name} (类名: {target.ClassName})")
        return target
    
    def is_valid_main_window(self, control) -> bool:
        try:
            return (control.NativeWindowHandle != 0 and 
                    control.ClassName and 
                    control.ClassName != "Shell_TrayWnd")
        except:
            return False
    
    def explore_control_structure(self, control: auto.Control, max_depth: int = 20,
                                 keywords: Optional[List[str]] = None, regex_mode: bool = False) -> Dict[str, Any]:
        control_info = self._get_control_info(control)
        
        if max_depth > 0:
            control_info["children"] = []
            try:
                for child in control.GetChildren():
                    child_info = self.explore_control_structure(
                        child, max_depth - 1, keywords, regex_mode)
                    if child_info:
                        control_info["children"].append(child_info)
            except:
                pass
        
        self.found_controls.append(control_info)
        return control_info
    
    def _get_control_info(self, control: auto.Control) -> Dict[str, Any]:
        info = {
            "name": "",
            "class_name": "",
            "control_type": "Unknown",
            "automation_id": "",
            "is_enabled": False,
            "is_visible": False,
            "process_id": 0,
            "runtime_id": "",
            "coordinates": None,
            "children": []
        }
        
        for attr in ['Name', 'ClassName', 'ControlTypeName', 
                     'AutomationId', 'IsEnabled', 'IsVisible',
                     'ProcessId', 'RuntimeId']:
            try:
                val = getattr(control, attr)
                key = attr.lower()
                if key == 'controltypename':
                    key = 'control_type'
                elif key == 'processid':
                    key = 'process_id'
                elif key == 'runtimeid':
                    key = 'runtime_id'
                info[key] = val
            except:
                pass
        
        try:
            rect = control.BoundingRectangle
            info["coordinates"] = {
                "left": rect.left,
                "top": rect.top,
                "right": rect.right,
                "bottom": rect.bottom,
                "width": rect.width(),
                "height": rect.height()
            }
        except:
            pass
        
        return info

class WindowSelectionDialog(QDialog):
    def __init__(self, windows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("选择窗口")
        self.setGeometry(200, 200, 400, 300)

        layout = QVBoxLayout(self)

        self.list_widget = QListWidget(self)
        for window in windows:
            item = QListWidgetItem(f"{window['name']} ({window['class_name']})")
            item.setData(Qt.UserRole, window)
            self.list_widget.addItem(item)
        layout.addWidget(self.list_widget)

        button_box = QHBoxLayout()
        self.ok_button = QPushButton("确定", self)
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("取消", self)
        self.cancel_button.clicked.connect(self.reject)
        button_box.addWidget(self.ok_button)
        button_box.addWidget(self.cancel_button)
        layout.addLayout(button_box)

    def get_selected_window(self):
        selected_item = self.list_widget.currentItem()
        if selected_item:
            return selected_item.data(Qt.UserRole)
        return None

class GraphicView(QGraphicsView):
    def __init__(self, main_window, parent=None):  # 接收 main_window 参数
        super().__init__(parent)
        self.main_window = main_window  # 保存 main_window 的引用
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.controls = []
        self.highlighted_item = None
        self.init_ui()
    
    def init_ui(self):
        self.setRenderHint(QPainter.Antialiasing)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setInteractive(True)
        
        # 控件类型颜色映射
        self.type_colors = {
            "Button": QColor(65, 105, 225),   # 蓝色
            "Edit": QColor(34, 139, 34),      # 绿色
            "Menu": QColor(220, 20, 60),      # 红色
            "Window": QColor(128, 0, 128),    # 紫色
            "Pane": QColor(255, 165, 0),      # 橙色
            "Text": QColor(0, 139, 139),      # 青色
            "Other": QColor(128, 128, 128)    # 灰色
        }
    
    def draw_controls(self, controls):
        self.scene.clear()
        self.controls = controls
        self.highlighted_item = None
        
        # 计算整体边界
        min_x, min_y = float('inf'), float('inf')
        max_x, max_y = -float('inf'), -float('inf')
        
        for control in controls:
            if 'coordinates' not in control:
                continue
                
            rect = control['coordinates']
            min_x = min(min_x, rect['left'])
            min_y = min(min_y, rect['top'])
            max_x = max(max_x, rect['right'])
            max_y = max(max_y, rect['bottom'])
            
            # 绘制控件
            self.draw_control(control)
        
        # 设置场景边界
        if min_x != float('inf'):
            margin = 20
            self.scene.setSceneRect(min_x-margin, min_y-margin, 
                                  (max_x-min_x)+2*margin, (max_y-min_y)+2*margin)
        
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)
    
    def draw_control(self, control):
        rect = control['coordinates']
        ctrl_type = control.get('control_type', 'Other')
        color = self.type_colors.get(ctrl_type.split(' ')[0], self.type_colors['Other'])
        
        # 绘制矩形
        item = QGraphicsRectItem(
            rect['left'], rect['top'],
            rect['width'], rect['height'])
        
        item.setPen(QPen(color, 2))
        item.setBrush(QBrush(color.lighter(180)))
        item.setData(0, control)  # 存储控件数据

        # 初始化文本变量
        text = None

        # 添加标签
        if rect['width'] > 30 and rect['height'] > 15 and control.get('name', '').strip():
            text = QGraphicsTextItem(control.get('name', '')[:20])
            text.setPos(rect['left'] + 2, rect['top'] + 2)
            text.setDefaultTextColor(Qt.black)
            text.setData(0, control)
            
            # 确保文本不超出矩形
            text_width = text.boundingRect().width()
            if text_width > rect['width'] - 4:
                text.setScale((rect['width'] - 4) / text_width)
        
        self.scene.addItem(item)
        if text:  # 确保文本已定义
            self.scene.addItem(text)
    
    def highlight_control(self, control):
        # 清除之前的高亮
        if self.highlighted_item:
            self.highlighted_item.setPen(QPen(Qt.black, 1))
            self.highlighted_item.setZValue(0)
        
        # 查找并高亮新控件
        for item in self.scene.items():
            if isinstance(item, QGraphicsRectItem) and item.data(0) == control:
                item.setPen(QPen(Qt.yellow, 3))
                item.setZValue(1)
                self.highlighted_item = item
                self.centerOn(item)
                break
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            item = self.itemAt(event.pos())
            if isinstance(item, (QGraphicsRectItem, QGraphicsTextItem)):
                control = item.data(0)
                self.main_window.sync_views('graphic', control)  # 使用 main_window 调用 sync_views
        super().mousePressEvent(event)

class FlatView(QTabWidget):
    def __init__(self, main_window, parent=None):   # 接收 main_window 参数
        super().__init__(parent)
        self.main_window = main_window  # 保存 main_window 的引用
        self.tables = {}
        self.type_map = {
            "Button": [],
            "Edit": [],
            "Menu": [],
            "Window": [],
            "Pane": [],
            "Text": [],
            "Other": []
        }
        self.init_ui()
    
    def init_ui(self):
        self.setTabsClosable(False)
        
        # 为每种类型创建表格
        for type_name in self.type_map:
            table = QTableWidget()
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(["名称", "类名", "ID", "坐标", "状态"])
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
            table.verticalHeader().setVisible(False)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.SingleSelection)
            table.itemSelectionChanged.connect(self.on_table_selection_changed)
            
            self.tables[type_name] = table
            self.addTab(table, type_name)
    
    def update_data(self, controls):
        # 清空现有数据
        for table in self.tables.values():
            table.setRowCount(0)
            table.clearContents()
        
        # 重新分类控件
        for control in controls:
            ctrl_type = control.get('control_type', 'Other').split(' ')[0]
            target_type = ctrl_type if ctrl_type in self.type_map else 'Other'
            self.add_control_to_table(target_type, control)
    
    def add_control_to_table(self, type_name, control):
        table = self.tables[type_name]
        row = table.rowCount()
        table.insertRow(row)
        
        # 名称
        name_item = QTableWidgetItem(control.get('name', ''))
        name_item.setData(Qt.UserRole, control)
        table.setItem(row, 0, name_item)
        
        # 类名
        table.setItem(row, 1, QTableWidgetItem(control.get('class_name', '')))
        
        # ID
        table.setItem(row, 2, QTableWidgetItem(control.get('automation_id', '')))
        
        # 坐标
        coords = ""
        if 'coordinates' in control:
            c = control['coordinates']
            coords = f"({c['left']},{c['top']})-({c['right']},{c['bottom']})"
        table.setItem(row, 3, QTableWidgetItem(coords))
        
        # 状态
        state = []
        if control.get('is_enabled', False):
            state.append("启用")
        if control.get('is_visible', False):
            state.append("可见")
        table.setItem(row, 4, QTableWidgetItem(", ".join(state)))
    
    def on_table_selection_changed(self):
        selected_items = self.currentWidget().selectedItems()
        if not selected_items:
            return
        
        control = selected_items[0].data(Qt.UserRole)
        self.main_window.sync_views('flat', control)  # 使用 main_window 调用 sync_views
    
    def scroll_to_control(self, control):
        for type_name, table in self.tables.items():
            for row in range(table.rowCount()):
                if table.item(row, 0).data(Qt.UserRole) == control:
                    self.setCurrentWidget(table)
                    table.selectRow(row)
                    table.scrollToItem(table.item(row, 0))
                    return

class GUIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scanner = GUIScanner()
        self.current_structure = None
        self.init_ui()

         # 在启动时列出所有窗口并选择
        self.select_window_on_startup()

    def select_window_on_startup(self):
        windows = self.scanner.list_all_windows()
        if not windows:
            QMessageBox.warning(self, "警告", "未找到任何窗口！")
            return

        dialog = WindowSelectionDialog(windows, self)
        if dialog.exec_() == QDialog.Accepted:
            selected_window = dialog.get_selected_window()
            if selected_window:
                self.title_input.setText(selected_window["name"])
                self.class_input.setText(selected_window["class_name"])

    def init_ui(self):
        self.setWindowTitle("高级GUI结构探查工具 - 多视图版")
        self.setGeometry(100, 100, 1400, 900)
        
        # 创建菜单栏
        self.create_menu()
        
        # 创建工具栏
        self.create_toolbar()
        
        # 主控件
        self.create_search_panel()
        self.create_display_panels()
        self.create_status_bar()
        
        # 主布局
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        main_layout.addWidget(self.search_panel)
        main_layout.addWidget(self.view_tabs)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
        # 初始化视图
        self.current_view = 'tree'
        self.show_tree_view()
    
    def create_menu(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu('文件')
        export_action = file_menu.addAction('导出JSON')
        export_action.triggered.connect(self.export_to_json)
        file_menu.addSeparator()
        exit_action = file_menu.addAction('退出')
        exit_action.triggered.connect(self.close)
        
        # 视图菜单
        view_menu = menubar.addMenu('视图')
        self.tree_view_action = view_menu.addAction('树形视图')
        self.tree_view_action.triggered.connect(self.show_tree_view)
        self.flat_view_action = view_menu.addAction('平铺视图')
        self.flat_view_action.triggered.connect(self.show_flat_view)
        self.graphic_view_action = view_menu.addAction('图形视图')
        self.graphic_view_action.triggered.connect(self.show_graphic_view)
        
        # 选项菜单
        option_menu = menubar.addMenu('选项')
        self.show_details_action = option_menu.addAction('显示详细信息')
        self.show_details_action.setCheckable(True)
        self.show_details_action.setChecked(True)
        self.show_coords_action = option_menu.addAction('显示坐标信息')
        self.show_coords_action.setCheckable(True)
        self.show_coords_action.setChecked(True)
    
    def create_toolbar(self):
        toolbar = QToolBar("主工具栏")
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        
        # 视图切换按钮
        self.tree_view_btn = QAction(QIcon.fromTheme('view-list-tree'), "树形视图", self)
        self.tree_view_btn.triggered.connect(self.show_tree_view)
        toolbar.addAction(self.tree_view_btn)
        
        self.flat_view_btn = QAction(QIcon.fromTheme('view-list-details'), "平铺视图", self)
        self.flat_view_btn.triggered.connect(self.show_flat_view)
        toolbar.addAction(self.flat_view_btn)
        
        self.graphic_view_btn = QAction(QIcon.fromTheme('view-list-icons'), "图形视图", self)
        self.graphic_view_btn.triggered.connect(self.show_graphic_view)
        toolbar.addAction(self.graphic_view_btn)
        
        toolbar.addSeparator()
        
        # 刷新按钮
        refresh_btn = QAction(QIcon.fromTheme('view-refresh'), "刷新", self)
        refresh_btn.triggered.connect(self.refresh_view)
        toolbar.addAction(refresh_btn)
    
    def create_search_panel(self):
        self.search_panel = QWidget()
        search_layout = QVBoxLayout()
        
        # 第一行：标题和类名
        row1 = QWidget()
        row1_layout = QHBoxLayout()
        
        self.title_input = QLineEdit()
        self.title_input.setPlaceholderText("输入窗口标题或留空显示所有")
        row1_layout.addWidget(QLabel("窗口标题:"))
        row1_layout.addWidget(self.title_input)
        
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("可选类名")
        row1_layout.addWidget(QLabel("类名:"))
        row1_layout.addWidget(self.class_input)
        
        row1.setLayout(row1_layout)
        
        # 第二行：选项和按钮
        row2 = QWidget()
        row2_layout = QHBoxLayout()
        
        self.fuzzy_check = QCheckBox("模糊匹配")
        self.regex_check = QCheckBox("正则匹配")
        self.scan_button = QPushButton("开始扫描")
        self.scan_button.clicked.connect(self.start_scanning)
        
        row2_layout.addWidget(self.fuzzy_check)
        row2_layout.addWidget(self.regex_check)
        row2_layout.addStretch()
        row2_layout.addWidget(self.scan_button)
        
        row2.setLayout(row2_layout)
        
        search_layout.addWidget(row1)
        search_layout.addWidget(row2)
        self.search_panel.setLayout(search_layout)
    
    def create_display_panels(self):
        # 创建视图容器
        self.view_tabs = QTabWidget()
        self.view_tabs.setTabsClosable(False)
        
        # 树形视图
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["控件类型", "名称", "类名"])
        self.tree_widget.setColumnWidth(0, 200)
        self.tree_widget.itemSelectionChanged.connect(
            lambda: self.sync_views('tree'))
        
        # 平铺视图
        self.flat_view = FlatView(self)  # 传递 self
        
        # 图形视图
        self.graphic_view = GraphicView(self)  # 传递 self
        self.graphic_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.graphic_view.customContextMenuRequested.connect(
            self.show_graphic_context_menu)
        
        # 属性表格
        self.property_table = QTableWidget()
        self.property_table.setColumnCount(2)
        self.property_table.setHorizontalHeaderLabels(["属性", "值"])
        self.property_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.property_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.property_table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 创建分割器布局
        self.tree_splitter = QSplitter(Qt.Horizontal)
        self.tree_splitter.addWidget(self.tree_widget)
        self.tree_splitter.addWidget(self.property_table)
        self.tree_splitter.setSizes([400, 600])
        
        self.flat_splitter = QSplitter(Qt.Horizontal)
        self.flat_splitter.addWidget(self.flat_view)
        self.flat_splitter.addWidget(self.property_table)
        self.flat_splitter.setSizes([400, 600])
        
        self.graphic_splitter = QSplitter(Qt.Horizontal)
        self.graphic_splitter.addWidget(self.graphic_view)
        self.graphic_splitter.addWidget(self.property_table)
        self.graphic_splitter.setSizes([400, 600])
        
        # 添加到标签页
        self.view_tabs.addTab(self.tree_splitter, "树形视图")
        self.view_tabs.addTab(self.flat_splitter, "平铺视图")
        self.view_tabs.addTab(self.graphic_splitter, "图形视图")
    
    def create_status_bar(self):
        self.status_bar = self.statusBar()
        self.status_label = QLabel("就绪")
        self.status_bar.addWidget(self.status_label)
        
        self.scan_count_label = QLabel("扫描控件: 0")
        self.status_bar.addPermanentWidget(self.scan_count_label)
        
        self.time_label = QLabel("时间: 0ms")
        self.status_bar.addPermanentWidget(self.time_label)
    
    def show_tree_view(self):
        self.current_view = 'tree'
        self.view_tabs.setCurrentIndex(0)
        self.update_view_actions()
    
    def show_flat_view(self):
        self.current_view = 'flat'
        self.view_tabs.setCurrentIndex(1)
        self.update_view_actions()
    
    def show_graphic_view(self):
        self.current_view = 'graphic'
        self.view_tabs.setCurrentIndex(2)
        self.update_view_actions()
        
        # 如果已经有数据，重新绘制图形视图
        if self.current_structure:
            self.graphic_view.draw_controls(self.scanner.found_controls)
    
    def update_view_actions(self):
        self.tree_view_action.setChecked(self.current_view == 'tree')
        self.flat_view_action.setChecked(self.current_view == 'flat')
        self.graphic_view_action.setChecked(self.current_view == 'graphic')
        
        self.tree_view_btn.setChecked(self.current_view == 'tree')
        self.flat_view_btn.setChecked(self.current_view == 'flat')
        self.graphic_view_btn.setChecked(self.current_view == 'graphic')
    
    def refresh_view(self):
        if self.current_structure:
            if self.current_view == 'tree':
                self.display_results(self.current_structure)
            elif self.current_view == 'flat':
                self.flat_view.update_data(self.scanner.found_controls)
            elif self.current_view == 'graphic':
                self.graphic_view.draw_controls(self.scanner.found_controls)
    
    def start_scanning(self):
        title = self.title_input.text().strip()
        class_name = self.class_input.text().strip() or None
        fuzzy = self.fuzzy_check.isChecked()
        regex = self.regex_check.isChecked()
        
        self.status_label.setText("正在扫描...")
        QApplication.processEvents()
        
        self.scan_thread = ScannerThread(
            self.scanner, title, class_name, fuzzy, regex)
        self.scan_thread.scan_finished.connect(self.display_results)
        self.scan_thread.start()
    
    def display_results(self, structure):
        start_time = time.time()
        
        self.current_structure = structure
        self.tree_widget.clear()
        self.property_table.setRowCount(0)
        
        if structure:
            # 更新树形视图
            root_item = self.create_tree_item(structure)
            self.tree_widget.addTopLevelItem(root_item)
            self.tree_widget.expandAll()
            
            # 更新平铺视图
            self.flat_view.update_data(self.scanner.found_controls)
            
            # 更新图形视图
            self.graphic_view.draw_controls(self.scanner.found_controls)
            
            # 更新状态
            self.scan_count_label.setText(f"扫描控件: {len(self.scanner.found_controls)}")
            elapsed = int((time.time() - start_time) * 1000)
            self.time_label.setText(f"时间: {elapsed}ms")
            self.status_label.setText("扫描完成")
        else:
            self.status_label.setText("未找到匹配窗口")
    
    def create_tree_item(self, control_info):
        item = QTreeWidgetItem([
            control_info.get('control_type', ''),
            control_info.get('name', ''),
            control_info.get('class_name', '')
        ])
        item.setData(0, Qt.UserRole, control_info)
        
        for child in control_info.get('children', []):
            child_item = self.create_tree_item(child)
            item.addChild(child_item)
            
        return item
    
    def sync_views(self, source_view, control=None):
        if not control:
            # 从当前视图获取选中的控件
            if source_view == 'tree':
                selected_items = self.tree_widget.selectedItems()
                if selected_items:
                    control = selected_items[0].data(0, Qt.UserRole)
            elif source_view == 'flat':
                selected_items = self.flat_view.currentWidget().selectedItems()
                if selected_items:
                    control = selected_items[0].data(Qt.UserRole)
            elif source_view == 'graphic':
                return  # 图形视图的同步在鼠标事件中处理
        
        if not control:
            return
        
        # 更新属性表格
        self.update_property_table(control)
        
        # 同步其他视图
        if source_view != 'tree':
            self.highlight_in_tree_view(control)
        
        if source_view != 'flat':
            self.flat_view.scroll_to_control(control)
        
        if source_view != 'graphic':
            self.graphic_view.highlight_control(control)
    
    def highlight_in_tree_view(self, control):
        # 递归查找树中的对应项
        def find_item(item):
            if item.data(0, Qt.UserRole) == control:
                return item
            for i in range(item.childCount()):
                found = find_item(item.child(i))
                if found:
                    return found
            return None
        
        root = self.tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            item = find_item(root.child(i))
            if item:
                self.tree_widget.setCurrentItem(item)
                self.tree_widget.scrollToItem(item)
                break
    
    def update_property_table(self, control_info):
        self.property_table.setRowCount(0)
        
        # 添加基本属性
        self.add_property_row("名称", control_info.get('name', ''))
        self.add_property_row("类名", control_info.get('class_name', ''))
        self.add_property_row("控件类型", control_info.get('control_type', ''))
        self.add_property_row("自动化ID", control_info.get('automation_id', ''))
        self.add_property_row("进程ID", str(control_info.get('process_id', '')))
        self.add_property_row("是否启用", "是" if control_info.get('is_enabled', False) else "否")
        self.add_property_row("是否可见", "是" if control_info.get('is_visible', False) else "否")
        
        # 添加坐标信息
        if self.show_coords_action.isChecked() and control_info.get('coordinates'):
            coords = control_info['coordinates']
            self.add_property_row("位置", f"({coords['left']}, {coords['top']})")
            self.add_property_row("大小", f"{coords['width']}x{coords['height']}")
        
        # 添加其他属性
        if self.show_details_action.isChecked():
            for key, value in control_info.items():
                if key not in ['name', 'class_name', 'control_type', 'automation_id', 
                              'process_id', 'is_enabled', 'is_visible', 'coordinates', 'children']:
                    self.add_property_row(key, str(value))
    
    def add_property_row(self, name, value):
        row = self.property_table.rowCount()
        self.property_table.insertRow(row)
        
        name_item = QTableWidgetItem(name)
        value_item = QTableWidgetItem(value)
        
        self.property_table.setItem(row, 0, name_item)
        self.property_table.setItem(row, 1, value_item)
    
    def show_graphic_context_menu(self, pos):
        menu = QMenu()
        
        zoom_in = menu.addAction("放大")
        zoom_out = menu.addAction("缩小")
        fit_view = menu.addAction("适应窗口")
        menu.addSeparator()
        refresh = menu.addAction("刷新视图")
        
        action = menu.exec_(self.graphic_view.mapToGlobal(pos))
        if action == zoom_in:
            self.graphic_view.scale(1.2, 1.2)
        elif action == zoom_out:
            self.graphic_view.scale(0.8, 0.8)
        elif action == fit_view:
            self.graphic_view.fitInView(self.graphic_view.scene.itemsBoundingRect(), 
                                      Qt.KeepAspectRatio)
        elif action == refresh:
            self.graphic_view.draw_controls(self.scanner.found_controls)
    
    def export_to_json(self):
        if not hasattr(self.scanner, 'found_controls') or not self.scanner.found_controls:
            self.status_label.setText("没有可导出的数据")
            return
            
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "导出JSON文件", "", "JSON文件 (*.json)", options=options)
            
        if file_name:
            if not file_name.endswith('.json'):
                file_name += '.json'
                
            data = {
                "window_title": self.title_input.text(),
                "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "controls": self.scanner.found_controls
            }
            
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.status_label.setText(f"成功导出到: {file_name}")
            except Exception as e:
                self.status_label.setText(f"导出失败: {str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUIMainWindow()
    window.show()
    sys.exit(app.exec_())