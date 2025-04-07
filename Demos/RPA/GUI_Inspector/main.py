import sys
import json
import re
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTreeWidget, QTreeWidgetItem,
                            QTableWidget, QTableWidgetItem, QSplitter, QVBoxLayout,
                            QHBoxLayout, QWidget, QLabel, QLineEdit, QPushButton,
                            QCheckBox, QTabWidget, QToolBar, QAction, QGraphicsView,
                            QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem,
                            QFileDialog, QMenu, QStatusBar, QHeaderView, QDialog, 
                            QListWidget, QListWidgetItem,QMessageBox,QInputDialog, 
                            QTextEdit,QShortcut,QStyledItemDelegate,QSplitter, QTextEdit,
                            QStyle,QGroupBox,QComboBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF,QTimer,QSize
from PyQt5.QtGui import QColor, QPen, QBrush, QPainter, QIcon, QClipboard, QKeySequence,QPalette
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
        
        self.found_controls.append(control_info)  # 保持填充 found_controls
        return control_info  # 返回当前控件的信息字典
    
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
        self.setup_zoom()  # 新增：初始化缩放控制
    
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

        # 添加图例
        self.draw_legend()  # 新增：初始化时绘制图例
    
    # 新增：缩放控制设置
    def setup_zoom(self):
        self.zoom_factor = 1.0
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
    
    # 新增：绘制图例
    def draw_legend(self):
        legend = QGraphicsRectItem(10, 10, 150, 20 + 30 * len(self.type_colors))
        legend.setBrush(QBrush(Qt.white))
        legend.setOpacity(0.8)  # 半透明效果
        self.scene.addItem(legend)
        
        for i, (type_name, color) in enumerate(self.type_colors.items()):
            # 绘制颜色示例
            color_rect = QGraphicsRectItem(20, 20 + 30 * i, 20, 20)
            color_rect.setBrush(QBrush(color))
            color_rect.setPen(QPen(Qt.black, 1))
            self.scene.addItem(color_rect)
            
            # 添加类型文本
            text = QGraphicsTextItem(type_name)
            text.setPos(50, 20 + 30 * i)
            self.scene.addItem(text)
    
    # 新增：滚轮缩放事件
    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        
        # 保存当前缩放中心
        old_pos = self.mapToScene(event.pos())
        
        # 缩放
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
            self.zoom_factor *= zoom_in_factor
        else:
            self.scale(zoom_out_factor, zoom_out_factor)
            self.zoom_factor *= zoom_out_factor
        
        # 保持缩放中心
        new_pos = self.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())
    
    # 新增：重置缩放
    def reset_zoom(self):
        self.resetTransform()
        self.zoom_factor = 1.0

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
        item.setData(0, control)
        
        # 添加类型标签
        type_text = QGraphicsTextItem(ctrl_type.split(' ')[0][:10])  # 限制类型名称长度
        type_text.setPos(rect['left'] + 2, rect['top'] + 2)
        type_text.setDefaultTextColor(Qt.black)
        type_text.setData(0, control)
        
        # 添加名称标签（如果空间足够）
        name = control.get('name', '')
        if name and rect['width'] > 80 and rect['height'] > 40:
            name_text = QGraphicsTextItem(name[:15])  # 限制名称长度
            name_text.setPos(rect['left'] + 2, rect['top'] + 20)
            name_text.setDefaultTextColor(Qt.darkBlue)
            name_text.setData(0, control)
        
        self.scene.addItem(item)
        self.scene.addItem(type_text)
        if name and rect['width'] > 80 and rect['height'] > 40:
            self.scene.addItem(name_text)
    
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

# 在TreeView中添加辅助线样式
class TreeView(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
            /* 使用Unicode字符代替图片 */
            QTreeView::branch:has-siblings:!adjoins-item {
                border-image: none;
                background: none;
            }
            QTreeView::branch:has-siblings:adjoins-item {
                border-image: none;
                background: none;
            }
            QTreeView::branch:!has-children:!has-siblings:adjoins-item {
                border-image: none;
                background: none;
            }
            /* 使用纯CSS箭头 */
            QTreeView::branch:open:has-children {
                image: url(none);
                border-image: none;
            }
            QTreeView::branch:closed:has-children {
                image: url(none);
                border-image: none;
            }
            /* 连接线样式 */
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {
                border-image: none;
                image: none;
            }
        """)
        
        # 关键渲染设置
        self.setIndentation(15)  # 缩进宽度
        self.setAnimated(True)
        self.setUniformRowHeights(True)  # 统一行高
        self.setAllColumnsShowFocus(True)
        self.setRootIsDecorated(True)
        
        # 使用QPen绘制连接线
        self.setItemDelegate(TreeItemDelegate())

class TreeItemDelegate(QStyledItemDelegate):
    def paint(self, painter, option, index):
        super().paint(painter, option, index)
        
        # 绘制自定义连接线
        if index.parent().isValid():  # 不是根节点
            painter.save()
            pen = QPen(QColor("#c0c0c0"), 1, Qt.SolidLine)
            painter.setPen(pen)
            
            rect = option.rect
            x = rect.left() + 7
            top_y = rect.top()
            bottom_y = rect.bottom()
            mid_y = rect.center().y()
            
            # 垂直线
            painter.drawLine(x, top_y, x, bottom_y)
            
            # 水平连接线
            painter.drawLine(x, mid_y, x + 10, mid_y)
            
            # 展开/折叠标记
            if option.state & QStyle.State_Children:
                center = QPointF(x, mid_y)
                radius = 3
                painter.drawEllipse(center, radius, radius)
                if option.state & QStyle.State_Open:
                    painter.drawText(center + QPointF(-2, 3), "-")
                else:
                    painter.drawText(center + QPointF(-2, 3), "+")
            
            painter.restore()

class PropertyPanel(QWidget):
    """增强版属性面板"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(300)
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        
        # 控件基本信息组
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
        """更新属性显示"""
        self.table.setRowCount(0)
        self.xpath_edit.clear()
        
        if not control_info:
            return
            
        # 生成XPath
        xpath = XPathGenerator.generate_full_xpath(control_info)
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
            
            # 设置字体和样式
            name_item.setFlags(Qt.ItemIsEnabled)
            value_item.setFlags(Qt.ItemIsEnabled)
            
            if "✅" in str(value):
                value_item.setForeground(QColor(0, 128, 0))
            elif "❌" in str(value):
                value_item.setForeground(QColor(220, 0, 0))
            
            self.table.setItem(row, 0, name_item)
            self.table.setItem(row, 1, value_item)
    
    def copy_xpath(self):
        """复制XPath到剪贴板"""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.xpath_edit.toPlainText())
        QMessageBox.information(self, "成功", "XPath已复制到剪贴板")

class FlatView(QTabWidget):
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window
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
            
            # 设置列宽策略
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Interactive)  # 名称列可调整
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)  # 类名自适应
            table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)  # ID自适应
            table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)  # 坐标拉伸
            table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeToContents)  # 状态自适应
            
            # 设置初始列宽
            table.setColumnWidth(0, 200)  # 名称列初始宽度
            table.setColumnWidth(1, 120)  # 类名列初始宽度
            table.setColumnWidth(2, 80)   # ID列初始宽度
            
            table.verticalHeader().setVisible(False)
            table.setSelectionBehavior(QTableWidget.SelectRows)
            table.setSelectionMode(QTableWidget.SingleSelection)
            table.itemSelectionChanged.connect(self.on_table_selection_changed)
            
            self.tables[type_name] = table
            self.addTab(table, type_name)
    
    def update_data(self, root_control):
        # 清空现有数据
        for table in self.tables.values():
            table.setRowCount(0)
            table.clearContents()
        
        # 使用队列进行广度优先遍历
        from collections import deque
        queue = deque([root_control])
        
        while queue:
            control = queue.popleft()
            
            # 添加当前控件
            ctrl_type = control.get('control_type', 'Other').split(' ')[0]
            target_type = ctrl_type if ctrl_type in self.type_map else 'Other'
            self.add_control_to_table(target_type, control)
            
            # 将子控件加入队列
            for child in control.get('children', []):
                queue.append(child)
    
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
# XPath生成器类
class XPathGenerator:
    @staticmethod
    def generate_xpath(control_info, parent_path=''):
        """生成控件的XPath表达式"""
        control_type = control_info.get('control_type', 'Control')
        name = control_info.get('name', '')
        automation_id = control_info.get('automation_id', '')
        class_name = control_info.get('class_name', '')
        
        # 构建基本定位条件
        conditions = []
        if name:
            conditions.append(f"@Name='{name}'")
        if automation_id:
            conditions.append(f"@AutomationId='{automation_id}'")
        if class_name:
            conditions.append(f"@ClassName='{class_name}'")
        
        # 如果没有唯一标识，使用索引
        if not conditions:
            if parent_path:
                # 尝试从父路径中获取索引
                if '[' in parent_path:
                    base_path = parent_path.split('[')[0]
                    same_type = [c for c in control_info.get('parent', {}).get('children', []) 
                               if c.get('control_type') == control_type]
                    if len(same_type) > 1:
                        idx = same_type.index(control_info) + 1
                        conditions.append(f"position()={idx}")
        
        # 构建XPath
        element_name = control_type.replace(' ', '')
        condition_str = ' and '.join(conditions)
        xpath = f"//{element_name}"
        if condition_str:
            xpath = f"{xpath}[{condition_str}]"
        
        return xpath

    @staticmethod
    def generate_full_xpath(control_info, path_parts=None):
        """递归生成完整XPath路径"""
        if path_parts is None:
            path_parts = []
        
        xpath = XPathGenerator.generate_xpath(control_info)
        path_parts.insert(0, xpath)
        
        if 'parent' in control_info and control_info['parent']:
            return XPathGenerator.generate_full_xpath(control_info['parent'], path_parts)
        
        return '/'.join(path_parts)

class GUIMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.scanner = GUIScanner()
        self.current_structure = None
        self.init_ui()
        self.add_control_actions()  # 这行初始化操作功能
        self.select_window_on_startup() # 在启动时列出所有窗口并选择

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

    def apply_styles(self):
        """应用现代化样式表"""
        style = """
        /* 主窗口样式 */
        QMainWindow {
            background: #f5f7fa;
            font-family: 'Microsoft YaHei', Arial;
        }
        
        /* 工具栏样式 */
        QToolBar {
            background: #e1e5eb;
            border-bottom: 1px solid #d1d5db;
            padding: 4px;
            spacing: 5px;
        }
        
        /* 按钮样式 */
        QPushButton {
            background: #4b5563;
            color: white;
            border: none;
            padding: 5px 12px;
            border-radius: 4px;
            min-width: 80px;
        }
        
        QPushButton:hover {
            background: #374151;
        }
        
        /* 输入框样式 */
        QLineEdit, QComboBox {
            border: 1px solid #d1d5db;
            border-radius: 4px;
            padding: 5px;
            background: white;
        }
        
        /* 树形视图样式 */
        QTreeView {
            alternate-background-color: #f8fafc;
            outline: 0;
        }
        
        QTreeView::item {
            padding: 4px 0;
        }
        
        /* 选项卡样式 */
        QTabWidget::pane {
            border: 1px solid #d1d5db;
            top: -1px;
        }
        
        QTabBar::tab {
            background: #e5e7eb;
            padding: 8px 12px;
            border: 1px solid #d1d5db;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background: white;
            border-bottom-color: white;
        }
        
        /* 属性面板样式 */
        QTableWidget {
            gridline-color: #e5e7eb;
        }
        
        QHeaderView::section {
            background: #f3f4f6;
            padding: 5px;
        }
        """
        self.setStyleSheet(style)

    def init_ui(self):
        self.setWindowTitle("GUI Inspector Pro")
        self.setGeometry(100, 100, 1600, 900)
        # 主布局容器
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        
        # 主布局（垂直）
        main_layout = QVBoxLayout()
        main_widget.setLayout(main_layout)
        
        # 顶部工具栏
        self.create_search_toolbar()
        main_layout.addWidget(self.search_toolbar)
        
        # 中央工作区（水平分割）
        workspace = QSplitter(Qt.Horizontal)
        main_layout.addWidget(workspace)
        
        # 左侧面板（树形+属性）
        left_panel = QSplitter(Qt.Vertical)
        workspace.addWidget(left_panel)
        
        # 树形视图（带标题）
        tree_group = QGroupBox("控件层级")
        tree_layout = QVBoxLayout()
        self.tree_widget = TreeView()
        tree_layout.addWidget(self.tree_widget)
        tree_group.setLayout(tree_layout)
        left_panel.addWidget(tree_group)
        
        # 属性面板（带标题）
        prop_group = QGroupBox("属性详情")
        prop_layout = QVBoxLayout()
        self.property_panel = PropertyPanel()
        prop_layout.addWidget(self.property_panel)
        prop_group.setLayout(prop_layout)
        left_panel.addWidget(prop_group)
        
        # 右侧面板（选项卡视图）
        right_panel = QTabWidget()
        workspace.addWidget(right_panel)
        
        # 平铺视图
        flat_tab = QWidget()
        flat_layout = QVBoxLayout()
        self.flat_view = FlatView(self)
        flat_layout.addWidget(self.flat_view)
        flat_tab.setLayout(flat_layout)
        right_panel.addTab(flat_tab, "🖥️ 平铺视图")
        
        # 图形视图
        graphic_tab = QWidget()
        graphic_layout = QVBoxLayout()
        self.graphic_view = GraphicView()
        graphic_layout.addWidget(self.graphic_view)
        graphic_tab.setLayout(graphic_layout)
        right_panel.addTab(graphic_tab, "📊 图形视图")
        
        # 设置初始比例
        left_panel.setSizes([600, 300])
        workspace.setSizes([400, 800])
        
        # 应用样式
        self.apply_styles()
    
    def add_shortcuts(self):
        # 缩放快捷键
        zoom_in = QShortcut(QKeySequence("Ctrl++"), self)
        zoom_in.activated.connect(lambda: self.graphic_view.scale(1.25, 1.25))
        
        zoom_out = QShortcut(QKeySequence("Ctrl+-"), self)
        zoom_out.activated.connect(lambda: self.graphic_view.scale(0.8, 0.8))
        
        reset_zoom = QShortcut(QKeySequence("Ctrl+0"), self)
        reset_zoom.activated.connect(self.graphic_view.reset_zoom)

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
    
    def create_search_toolbar(self):
        """新版搜索工具栏"""
        self.search_toolbar = QToolBar("搜索工具栏")
        self.addToolBar(Qt.TopToolBarArea, self.search_toolbar)
        
        # 窗口选择下拉框
        self.window_combo = QComboBox()
        self.window_combo.setFixedWidth(250)
        self.search_toolbar.addWidget(QLabel("目标窗口:"))
        self.search_toolbar.addWidget(self.window_combo)
        
        # 搜索框
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("搜索控件...")
        self.search_input.setFixedWidth(200)
        self.search_toolbar.addWidget(self.search_input)
        
        # 选项按钮
        self.fuzzy_check = QCheckBox("模糊匹配")
        self.regex_check = QCheckBox("正则匹配")
        self.search_toolbar.addWidget(self.fuzzy_check)
        self.search_toolbar.addWidget(self.regex_check)
        
        # 操作按钮
        scan_btn = QPushButton("🔍 扫描")
        scan_btn.clicked.connect(self.start_scanning)
        refresh_btn = QPushButton("🔄 刷新")
        refresh_btn.clicked.connect(self.refresh_view)
        export_btn = QPushButton("💾 导出")
        export_btn.clicked.connect(self.export_to_json)
        
        self.search_toolbar.addWidget(scan_btn)
        self.search_toolbar.addWidget(refresh_btn)
        self.search_toolbar.addWidget(export_btn)
        
        # 初始化窗口列表
        self.update_window_list()

    def update_window_list(self):
        """更新窗口下拉列表"""
        self.window_combo.clear()
        windows = self.scanner.list_all_windows()
        for window in windows:
            self.window_combo.addItem(f"{window['name']} ({window['class_name']})", window)
    
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
        self.property_panel.table = QTableWidget()
        self.property_panel.table.setColumnCount(2)
        self.property_panel.table.setHorizontalHeaderLabels(["属性", "值"])
        self.property_panel.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.property_panel.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.property_panel.table.setEditTriggers(QTableWidget.NoEditTriggers)
        
        # 创建分割器布局
        self.tree_splitter = QSplitter(Qt.Horizontal)
        self.tree_splitter.addWidget(self.tree_widget)
        self.tree_splitter.addWidget(self.property_panel.table)
        self.tree_splitter.setSizes([400, 600])
        
        self.flat_splitter = QSplitter(Qt.Horizontal)
        self.flat_splitter.addWidget(self.flat_view)
        self.flat_splitter.addWidget(self.property_panel.table)
        self.flat_splitter.setSizes([400, 600])
        
        self.graphic_splitter = QSplitter(Qt.Horizontal)
        self.graphic_splitter.addWidget(self.graphic_view)
        self.graphic_splitter.addWidget(self.property_panel.table)
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
        try:
            start_time = time.time()
            
            # 重置所有视图
            self.tree_widget.clear()
            self.property_panel.table.setRowCount(0)
            self.scanner.found_controls = []
            
            if not structure:
                self.status_label.setText("未获取到有效窗口结构")
                return

            # 更新树形视图（带严格过滤）
            root_item = self.create_tree_item(structure)
            if root_item:
                self.tree_widget.addTopLevelItem(root_item)
                # 默认选中根节点并展开
                self.tree_widget.setCurrentItem(root_item)
                root_item.setExpanded(True)
            
            # 更新其他视图
            self.flat_view.update_data(structure)
            if hasattr(self.scanner, 'found_controls'):
                self.graphic_view.draw_controls(self.scanner.found_controls)
            
            # 更新状态栏
            elapsed_ms = int((time.time() - start_time) * 1000)
            control_count = len(self.scanner.found_controls) if hasattr(self.scanner, 'found_controls') else 0
            self.scan_count_label.setText(f"控件总数: {control_count}")
            self.time_label.setText(f"处理时间: {elapsed_ms}ms")
            self.status_label.setText("扫描完成")
            
        except Exception as e:
            print(f"[ERROR] 结果显示失败: {str(e)}")
            self.status_label.setText("结果显示错误")
    
    def _populate_tree(self, structure):
        """线程安全的数据填充方法"""
        try:
            root_item = self.create_tree_item(structure)
            if root_item:
                self.tree_widget.addTopLevelItem(root_item)
                self.tree_widget.expandAll()
        except Exception as e:
            print(f"树形视图加载错误: {e}")

    def create_tree_item(self, control_info):
        """创建树形项目（严格过滤空值）"""
        if not control_info or not isinstance(control_info, dict):
            return None

        # 检查是否为有效控件（至少有一个非空字段）
        required_fields = ['name', 'class_name', 'control_type', 'coordinates']
        has_valid_data = any(
            control_info.get(field) 
            for field in required_fields 
            if isinstance(control_info.get(field), (str, dict)) and control_info.get(field)
        )
        
        if not has_valid_data:
            return None

        # 创建项目
        item = QTreeWidgetItem()
        item.setText(0, control_info.get('control_type', 'Unknown')[:50])
        item.setText(1, control_info.get('name', '')[:100])
        item.setText(2, control_info.get('class_name', '')[:50])
        
        # 设置完整数据用于属性面板
        item.setData(0, Qt.UserRole, control_info)
        
        # 递归处理子项（严格过滤）
        valid_children = []
        for child in control_info.get('children', []):
            child_item = self.create_tree_item(child)
            if child_item:
                valid_children.append(child_item)
        
        if valid_children:
            item.addChildren(valid_children)
            # 默认展开非空节点
            item.setExpanded(True)
        
        return item
    
    def sync_views(self, source_view, control=None):
        """同步所有视图的选中状态"""
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
                return
        
        if not control:
            return
        
        # 更新属性面板（替换原来的property_table更新）
        if hasattr(self, 'property_panel'):
            self.property_panel.update_properties(control)
        
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
        self.property_panel.table.setRowCount(0)
        
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
        row = self.property_panel.table.rowCount()
        self.property_panel.table.insertRow(row)
        
        name_item = QTableWidgetItem(name)
        value_item = QTableWidgetItem(value)
        
        self.property_panel.table.setItem(row, 0, name_item)
        self.property_panel.table.setItem(row, 1, value_item)
    
    def show_graphic_context_menu(self, pos):
        menu = QMenu()
        
        # 缩放控制
        zoom_in = menu.addAction("放大 (Ctrl++)")
        zoom_out = menu.addAction("缩小 (Ctrl+-)")
        reset_zoom = menu.addAction("重置缩放")
        
        # 视图操作
        fit_view = menu.addAction("适应窗口")
        menu.addSeparator()
        refresh = menu.addAction("刷新视图")
        
        action = menu.exec_(self.graphic_view.mapToGlobal(pos))
        
        if action == zoom_in:
            self.graphic_view.scale(1.25, 1.25)
        elif action == zoom_out:
            self.graphic_view.scale(0.8, 0.8)
        elif action == reset_zoom:
            self.graphic_view.reset_zoom()
        elif action == fit_view:
            self.graphic_view.fitInView(self.graphic_view.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
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

    def add_control_actions(self):
        """更新后的控件操作初始化方法"""
        # 为树形视图添加右键菜单
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(
            self.show_tree_context_menu)
        
        # 为属性面板添加右键菜单
        if hasattr(self, 'property_panel') and hasattr(self.property_panel, 'table'):
            self.property_panel.table.setContextMenuPolicy(Qt.CustomContextMenu)
            self.property_panel.table.customContextMenuRequested.connect(
                self.show_property_context_menu)

    def show_property_context_menu(self, pos):
        """属性面板的右键菜单"""
        # 获取当前选中的树形视图项
        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            return
        
        # 从树形视图获取控件信息（不再从property_table获取）
        control_info = selected_items[0].data(0, Qt.UserRole)
        
        menu = QMenu()
        click_action = menu.addAction("模拟点击")
        click_action.triggered.connect(lambda: self.operate_control(control_info, 'click'))
        
        if control_info.get('control_type', '') == 'Edit':
            input_action = menu.addAction("输入文本...")
            input_action.triggered.connect(lambda: self.show_input_dialog(control_info))
        
        menu.exec_(self.property_panel.table.mapToGlobal(pos))  # 注意改为property_panel.table

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

    # XPath相关方法
    def copy_xpath_to_clipboard(self, control_info):
        """复制XPath到剪贴板"""
        try:
            xpath = XPathGenerator.generate_full_xpath(control_info)
            clipboard = QApplication.clipboard()
            clipboard.setText(xpath)
            self.status_label.setText("XPath已复制到剪贴板")
        except Exception as e:
            self.status_label.setText(f"生成XPath失败: {str(e)}")

    def show_xpath_in_dialog(self, control_info):
        """在对话框中显示XPath"""
        xpath = XPathGenerator.generate_full_xpath(control_info)
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

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 设置全局样式
    app.setStyle("Fusion")
    palette = app.palette()
    palette.setColor(QPalette.Highlight, QColor(204, 224, 255))
    palette.setColor(QPalette.HighlightedText, Qt.black)
    app.setPalette(palette)
    
    window = GUIMainWindow()
    window.show()
    sys.exit(app.exec_())