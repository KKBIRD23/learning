import sys
import json
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem,
                            QTableWidget, QTableWidgetItem, QSplitter, QVBoxLayout,
                            QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton,
                            QCheckBox, QTabWidget, QToolBar, QAction, 
                            QFileDialog, QMenu, QStatusBar, QHeaderView, QDialog, 
                            QListWidget, QListWidgetItem,QMessageBox,QInputDialog, 
                            QTextEdit,QStyledItemDelegate,QGroupBox,QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer
from PyQt5.QtGui import QColor, QPen, QBrush, QPainter, QIcon, QClipboard, QKeySequence
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
    
    def explore_control_structure(self, control: auto.Control, max_depth: int = 20) -> Dict[str, Any]:
        """探索控件结构并返回结构化数据"""
        control_info = self._get_control_info(control)
        
        if max_depth > 0:
            control_info["children"] = []
            try:
                children = control.GetChildren()
                for child in children:
                    try:
                        child_info = self.explore_control_structure(child, max_depth - 1)
                        if child_info:
                            control_info["children"].append(child_info)
                    except Exception as e:
                        print(f"处理子控件时出错: {str(e)}")
                        continue
            except Exception as e:
                print(f"获取子控件时出错: {str(e)}")
        
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

class EnhancedTreeWidget(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["控件类型", "名称", "类名"])
        self.setColumnWidth(0, 200)
        self.setStyleSheet("""
            QTreeView {
                show-decoration-selected: 1;
                alternate-background-color: #f8f8f8;
                outline: 0;
            }
            QTreeView::item {
                border-left: 1px solid #e0e0e0;
                border-bottom: 1px solid #e0e0e0;
                padding: 2px;
                height: 22px;
            }
            QTreeView::branch {
                background: palette(base);
            }
        """)
        
        # 设置缩进和动画
        self.setIndentation(15)
        self.setAnimated(True)
        self.setUniformRowHeights(True)
        self.setAllColumnsShowFocus(True)
        self.setRootIsDecorated(True)

class PropertyPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # 基本信息组
        self.basic_group = QGroupBox("基本信息")
        self.basic_layout = QVBoxLayout()
        
        # 属性表格
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(["属性", "值"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        
        self.basic_layout.addWidget(self.table)
        self.basic_group.setLayout(self.basic_layout)
        
        # XPath组
        self.xpath_group = QGroupBox("XPath定位")
        self.xpath_layout = QVBoxLayout()
        
        self.xpath_edit = QTextEdit()
        self.xpath_edit.setReadOnly(True)
        self.xpath_edit.setMaximumHeight(80)
        self.xpath_edit.setStyleSheet("font-family: Consolas, 'Courier New';")
        
        self.copy_btn = QPushButton("复制到剪贴板")
        self.copy_btn.clicked.connect(self.copy_xpath)
        
        self.xpath_layout.addWidget(self.xpath_edit)
        self.xpath_layout.addWidget(self.copy_btn)
        self.xpath_group.setLayout(self.xpath_layout)
        
        # 添加到主布局
        self.layout.addWidget(self.basic_group)
        self.layout.addWidget(self.xpath_group)
        self.layout.setStretch(0, 2)
        self.layout.setStretch(1, 1)
    
    def update_properties(self, control_info):
        self.table.setRowCount(0)
        self.xpath_edit.clear()
        
        if not control_info:
            return
            
        # 生成XPath
        xpath = self.generate_xpath(control_info)
        self.xpath_edit.setPlainText(xpath)
        
        # 添加属性行
        properties = [
            ("控件类型", control_info.get('control_type', 'Unknown')),
            ("名称", control_info.get('name', '')),
            ("类名", control_info.get('class_name', '')),
            ("自动化ID", control_info.get('automation_id', '')),
            ("进程ID", str(control_info.get('process_id', ''))),
            ("是否启用", "✅" if control_info.get('is_enabled') else "❌"),
            ("是否可见", "✅" if control_info.get('is_visible') else "❌"),
        ]
        
        # 添加坐标信息
        if 'coordinates' in control_info:
            coord = control_info['coordinates']
            properties.extend([
                ("位置", f"X:{coord['left']} Y:{coord['top']}"),
                ("大小", f"{coord['width']}×{coord['height']}")
            ])
        
        # 填充表格
        self.table.setRowCount(len(properties))
        for row, (name, value) in enumerate(properties):
            name_item = QTableWidgetItem(name)
            value_item = QTableWidgetItem(str(value))
            
            name_item.setFlags(Qt.ItemIsEnabled)
            value_item.setFlags(Qt.ItemIsEnabled)
            
            if "✅" in str(value):
                value_item.setForeground(QColor(0, 128, 0))
            elif "❌" in str(value):
                value_item.setForeground(QColor(220, 0, 0))
            
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, value_item)
    
    def generate_xpath(self, control_info):
        """生成控件的XPath表达式"""
        control_type = control_info.get('control_type', 'Control')
        name = control_info.get('name', '')
        automation_id = control_info.get('automation_id', '')
        class_name = control_info.get('class_name', '')
        
        conditions = []
        if name:
            conditions.append(f"@Name='{name}'")
        if automation_id:
            conditions.append(f"@AutomationId='{automation_id}'")
        if class_name:
            conditions.append(f"@ClassName='{class_name}'")
        
        element_name = control_type.replace(' ', '')
        condition_str = ' and '.join(conditions)
        xpath = f"//{element_name}"
        if condition_str:
            xpath = f"{xpath}[{condition_str}]"
        
        return xpath
    
    def copy_xpath(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.xpath_edit.toPlainText())
        QMessageBox.information(self, "成功", "XPath已复制到剪贴板")

class FlatView(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.tables = {}
        self.type_map = {
            "Button": [], "Edit": [], "Menu": [], 
            "Window": [], "Pane": [], "Text": [], "Other": []
        }
        self.init_ui()
    
    def init_ui(self):
        self.setTabsClosable(False)
        
        for type_name in self.type_map:
            table = QTableWidget()
            table.setColumnCount(5)
            table.setHorizontalHeaderLabels(["名称", "类名", "ID", "坐标", "状态"])
            
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
            table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)
            
            table.setColumnWidth(0, 200)
            table.setColumnWidth(1, 120)
            table.setColumnWidth(2, 80)
            
            table.verticalHeader().setVisible(False)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.SingleSelection)
            
            self.tables[type_name] = table
            self.addTab(table, type_name)
    
    def update_data(self, root_control):
        # 清空现有数据
        for table in self.tables.values():
            table.setRowCount(0)
            table.clearContents()
        
        # 分类控件
        self._categorize_controls(root_control)
        
        # 填充表格
        for type_name, controls in self.type_map.items():
            table = self.tables[type_name]
            for control in controls:
                self._add_control_to_table(table, control)
    
    def _categorize_controls(self, control):
        """递归分类控件"""
        ctrl_type = control.get('control_type', 'Other').split(' ')[0]
        target_type = ctrl_type if ctrl_type in self.type_map else 'Other'
        self.type_map[target_type].append(control)
        
        for child in control.get('children', []):
            self._categorize_controls(child)
    
    def _add_control_to_table(self, table, control):
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

class GUIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scanner = GUIScanner()
        self.current_structure = None
        self.init_ui()
        self.add_control_actions()
        self.select_window_on_startup()

    def init_ui(self):
        self.setWindowTitle("GUI结构探查工具")
        self.setGeometry(100, 100, 1200, 800)

        # 创建菜单栏和工具栏
        self.create_menu()
        self.create_toolbar()

        # 主布局容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局（垂直）
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 顶部工具栏
        self.create_search_toolbar()
        main_layout.addWidget(self.search_toolbar)

        # 创建状态栏
        self.create_status_bar()
        
        # 中央工作区（水平分割）
        workspace = QSplitter(Qt.Horizontal)
        main_layout.addWidget(workspace)
        
        # 左侧面板（树形视图）
        self.tree_widget = EnhancedTreeWidget()
        self.tree_widget.itemSelectionChanged.connect(self.on_tree_selection_changed)
        workspace.addWidget(self.tree_widget)
        
        # 右侧面板（平铺视图和属性面板）
        right_panel = QSplitter(Qt.Vertical)
        workspace.addWidget(right_panel)
        
        # 平铺视图
        self.flat_view = FlatView()
        right_panel.addWidget(self.flat_view)
        
        # 属性面板
        self.property_panel = PropertyPanel()
        right_panel.addWidget(self.property_panel)
        
        # 设置初始比例
        workspace.setSizes([400, 800])
        right_panel.setSizes([500, 300])

    def create_search_toolbar(self):
        self.search_toolbar = QToolBar("搜索工具栏")
        self.addToolBar(Qt.TopToolBarArea, self.search_toolbar)
        
        # 窗口选择下拉框
        self.window_combo = QComboBox()
        self.window_combo.setFixedWidth(250)
        self.search_toolbar.addWidget(QLabel("目标窗口:"))
        self.search_toolbar.addWidget(self.window_combo)
        
        # 操作按钮
        scan_btn = QPushButton("扫描窗口")
        scan_btn.clicked.connect(self.start_scanning)
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self.refresh_view)
        export_btn = QPushButton("导出JSON")
        export_btn.clicked.connect(self.export_to_json)
        
        self.search_toolbar.addWidget(scan_btn)
        self.search_toolbar.addWidget(refresh_btn)
        self.search_toolbar.addWidget(export_btn)
        
        # 初始化窗口列表
        self.update_window_list()

    def create_status_bar(self):
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        self.status_label = QLabel("就绪")
        self.scan_count_label = QLabel("控件总数: 0")
        self.time_label = QLabel("处理时间: 0ms")
        
        self.status_bar.addWidget(self.status_label, 1)
        self.status_bar.addPermanentWidget(self.scan_count_label)
        self.status_bar.addPermanentWidget(self.time_label)

    def update_window_list(self):
        self.window_combo.clear()
        windows = self.scanner.list_all_windows()
        for window in windows:
            self.window_combo.addItem(f"{window['name']} ({window['class_name']})", window)

    def select_window_on_startup(self):
        windows = self.scanner.list_all_windows()
        if windows:
            dialog = WindowSelectionDialog(windows, self)
            if dialog.exec_() == QDialog.Accepted:
                selected_window = dialog.get_selected_window()
                if selected_window:
                    self.start_scanning(selected_window['name'], selected_window['class_name'])

    def start_scanning(self, title=None, class_name=None):
        if not title:
            window_data = self.window_combo.currentData()
            if not window_data:
                QMessageBox.warning(self, "警告", "请先选择目标窗口")
                return
            title = window_data['name']
            class_name = window_data['class_name']
        
        self.status_label.setText(f"正在扫描窗口: {title}...")
        QApplication.processEvents()
        
        self.scan_thread = ScannerThread(
            self.scanner, title, class_name, False, False)
        self.scan_thread.scan_finished.connect(self.display_results)
        self.scan_thread.start()

    def display_results(self, structure):
        try:
            start_time = time.time()
            
            # 重置所有视图
            self.tree_widget.clear()
            self.property_panel.table.setRowCount(0)
            self.scanner.found_controls = []
            
            if not structure:
                self.status_label.setText("未获取到有效窗口结构")
                return

            # 更新树形视图
            root_item = self.create_tree_item(structure)
            if root_item:
                self.tree_widget.addTopLevelItem(root_item)
                self.tree_widget.setCurrentItem(root_item)
                root_item.setExpanded(True)
            
            # 更新平铺视图
            self.flat_view.update_data(structure)
            
            # 更新状态栏
            elapsed_ms = int((time.time() - start_time) * 1000)
            control_count = self.count_controls(structure)
            self.scan_count_label.setText(f"控件总数: {control_count}")
            self.time_label.setText(f"处理时间: {elapsed_ms}ms")
            self.status_label.setText("扫描完成")
            
        except Exception as e:
            print(f"[ERROR] 结果显示失败: {str(e)}")
            self.status_label.setText("结果显示错误")

    def count_controls(self, control):
        """递归计算控件数量"""
        count = 1  # 当前控件
        for child in control.get('children', []):
            count += self.count_controls(child)
        return count

    def create_tree_item(self, control_info, parent_item=None):
        """递归创建树形结构"""
        if parent_item is None:
            parent_item = self.tree_widget.invisibleRootItem()
        
        item_text = f"{control_info.get('control_type', 'Unknown')} - {control_info.get('name', '')}"
        item = QTreeWidgetItem(parent_item, [item_text])
        item.setData(0, Qt.UserRole, control_info)
        
        for child in control_info.get('children', []):
            self.create_tree_item(child, item)
        
        return item

    def on_tree_selection_changed(self):
        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            return
        
        control = selected_items[0].data(0, Qt.UserRole)
        self.property_panel.update_properties(control)
        
        # 在平铺视图中高亮对应项
        self.highlight_in_flat_view(control)

    def highlight_in_flat_view(self, control):
        """在平铺视图中高亮指定控件"""
        for type_name, table in self.flat_view.tables.items():
            for row in range(table.rowCount()):
                if table.item(row, 0).data(Qt.UserRole) == control:
                    self.flat_view.setCurrentWidget(table)
                    table.selectRow(row)
                    table.scrollToItem(table.item(row, 0))
                    return

    def refresh_view(self):
        if self.current_structure:
            self.display_results(self.current_structure)

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

            window_data = self.window_combo.currentData()
            window_title = window_data['name'] if window_data else "未知窗口"
                
            data = {
                "window_title": window_title,
                "window_class": window_data['class_name'] if window_data else "",
                "scan_time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "controls": self.scanner.found_controls
            }
            
            try:
                with open(file_name, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                self.status_label.setText(f"成功导出到: {file_name}")
            except Exception as e:
                self.status_label.setText(f"导出失败: {str(e)}")

    def add_control_actions(self):
        """为控件添加操作"""
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
            self.show_tree_context_menu)

    def show_tree_context_menu(self, pos):
        item = self.tree_widget.itemAt(pos)
        if not item:
            return
        
        control_info = item.data(0, Qt.UserRole)
        menu = QMenu()
        
        # XPath相关操作
        xpath_action = menu.addAction("显示XPath")
        xpath_action.triggered.connect(lambda: self.show_xpath_in_dialog(control_info))
        
        copy_xpath = menu.addAction("复制XPath")
        copy_xpath.triggered.connect(lambda: self.copy_xpath_to_clipboard(control_info))
        
        menu.addSeparator()
        
        # 控件操作
        click_action = menu.addAction("模拟点击")
        click_action.triggered.connect(lambda: self.operate_control(control_info, 'click'))
        
        if control_info.get('control_type', '') == 'Edit':
            input_action = menu.addAction("输入文本...")
            input_action.triggered.connect(lambda: self.show_input_dialog(control_info))
        
        menu.addSeparator()
        
        # 刷新控件
        refresh_action = menu.addAction("刷新控件")
        refresh_action.triggered.connect(self.refresh_view)
        
        menu.exec_(self.tree_widget.mapToGlobal(pos))

    def operate_control(self, control_info, operation):
        try:
            control = self.find_control_by_info(control_info)
            if not control:
                self.status_label.setText("无法定位控件")
                return
                
            if operation == 'click':
                if hasattr(control, 'Click'):
                    control.Click()
                    self.status_label.setText("点击操作执行成功")
                else:
                    self.status_label.setText("该控件不支持点击操作")
        except Exception as e:
            self.status_label.setText(f"操作失败: {str(e)}")

    def show_input_dialog(self, control_info):
        text, ok = QInputDialog.getText(self, '输入文本', '请输入要输入的文本:')
        if ok and text:
            try:
                control = self.find_control_by_info(control_info)
                if hasattr(control, 'GetValuePattern'):
                    pattern = control.GetValuePattern()
                    pattern.SetValue(text)
                    self.status_label.setText("文本输入成功")
                elif hasattr(control, 'SendKeys'):
                    control.SendKeys(text)
                    self.status_label.setText("文本输入成功")
                else:
                    self.status_label.setText("该控件不支持文本输入")
            except Exception as e:
                self.status_label.setText(f"输入失败: {str(e)}")

    def find_control_by_info(self, control_info):
        """根据控件信息重新定位控件"""
        try:
            # 通过进程ID和运行时ID精确查找
            if control_info.get('process_id') and control_info.get('runtime_id'):
                control = auto.ControlFromHandle(control_info['process_id'], control_info['runtime_id'])
                if control.Exists():
                    return control
                    
            # 回退方法：通过属性和位置查找
            control = auto.WindowControl(
                searchDepth=1,
                Name=control_info.get('name', ''),
                ClassName=control_info.get('class_name', ''),
                AutomationId=control_info.get('automation_id', '')
            )
            return control if control.Exists() else None
        except:
            return None

    def copy_xpath_to_clipboard(self, control_info):
        """复制XPath到剪贴板"""
        try:
            xpath = self.property_panel.generate_xpath(control_info)
            clipboard = QApplication.clipboard()
            clipboard.setText(xpath)
            self.status_label.setText("XPath已复制到剪贴板")
        except Exception as e:
            self.status_label.setText(f"生成XPath失败: {str(e)}")

    def show_xpath_in_dialog(self, control_info):
        """在对话框中显示XPath"""
        xpath = self.property_panel.generate_xpath(control_info)
        dialog = QDialog(self)
        dialog.setWindowTitle("XPath定位表达式")
        layout = QVBoxLayout()
        
        text_edit = QTextEdit()
        text_edit.setPlainText(xpath)
        text_edit.setReadOnly(True)
        layout.addWidget(text_edit)
        
        button_box = QHBoxLayout()
        copy_btn = QPushButton("复制")
        copy_btn.clicked.connect(lambda: self.copy_xpath_to_clipboard(control_info))
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        button_box.addWidget(copy_btn)
        button_box.addWidget(close_btn)
        layout.addLayout(button_box)
        
        dialog.setLayout(layout)
        dialog.exec_()

    def create_menu(self):
        menubar = self.menuBar()
        
        # 文件菜单
        file_menu = menubar.addMenu("文件")
        
        exit_action = QAction("退出", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 视图菜单
        view_menu = menubar.addMenu("视图")
        
        refresh_action = QAction("刷新", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_view)
        view_menu.addAction(refresh_action)
        
        # 帮助菜单
        help_menu = menubar.addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def create_toolbar(self):
        toolbar = QToolBar("主工具栏")
        self.addToolBar(Qt.TopToolBarArea, toolbar)
        
        refresh_action = QAction(QIcon.fromTheme("view-refresh"), "刷新", self)
        refresh_action.setShortcut("F5")
        refresh_action.triggered.connect(self.refresh_view)
        toolbar.addAction(refresh_action)
        
        export_action = QAction(QIcon.fromTheme("document-save"), "导出", self)
        export_action.triggered.connect(self.export_to_json)
        toolbar.addAction(export_action)

    def show_about(self):
        QMessageBox.about(self, "关于", 
                         "GUI结构探查工具\n\n"
                         "版本: 1.0\n"
                         "用于分析和查看Windows应用程序的UI结构")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = GUIMainWindow()
    window.show()
    sys.exit(app.exec_())