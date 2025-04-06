import uiautomation as auto
import json
import csv
import time
from typing import Optional, Dict, Any, List
import os
import re
from datetime import datetime
from enum import Enum

class OutputFormat(Enum):
    JSON = 'json'
    CSV = 'csv'
    TXT = 'txt'

class GUIScanner:
    def __init__(self):
        self.scan_count = 0
        self.found_controls = []
        auto.uiautomation.SetGlobalSearchTimeout(10)

    def is_match(self, value: str, pattern: str, fuzzy: bool = False, regex: bool = False) -> bool:
        """辅助匹配函数：支持精确、模糊、正则"""
        if not value:
            return False
        if regex:
            return re.search(pattern, value, re.IGNORECASE) is not None
        elif fuzzy:
            return pattern.lower() in value.lower()
        else:
            return value == pattern
        
    def _is_system_tray_control(self, control) -> bool:
        """判断控件是否属于系统托盘"""
        try:
            class_name = control.ClassName
            return class_name in ["NotifyIconOverflowWindow", "Shell_TrayWnd"]
        except:
            return False
        
    def find_main_window(self, title: str, class_name: str = "", fuzzy=False, regex=False):
        print(f"\n正在查找窗口: '{title}'...")

        # 先尝试使用 auto.WindowControl 直接查找顶层窗口
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

        # 如果直接查找失败，则使用遍历方式，并增大扫描深度
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

        # 检查是否为托盘窗口
        if self._is_system_tray_control(target):
            choice = input("⚠️ 目标窗口可能是托盘图标，仍然继续？(y/n, 默认n): ").strip().lower()
            if choice != 'y':
                print("❌ 操作已取消。")
                return None

        print(f"\n✅ 找到目标窗口: {target.Name} (类名: {target.ClassName})")
        return target
    
    def is_valid_main_window(self, control) -> bool:
        """检查是否为有效主窗口"""
        try:
            # 此处放宽控制，允许 ControlTypeName 为 Window 或 Pane
            return (control.NativeWindowHandle != 0 and 
                    control.ClassName and 
                    control.ClassName != "Shell_TrayWnd")
        except:
            return False
    
    def find_and_activate_window(self, title: str, class_name: Optional[str] = None,
                                   fuzzy: bool = False, regex: bool = False,
                                   max_depth: int = 20) -> Optional[auto.Control]:
        """查找所有匹配窗口，支持选择并激活（兼容系统托盘控件）"""
        root = auto.GetRootControl()
        matches = []

        def collect_controls(control: auto.Control, current_depth: int = 0):
            if current_depth > max_depth:
                return
            try:
                name = control.Name or ""
                cls = control.ClassName or ""
            except:
                name = cls = ""

            if self.is_match(name, title, fuzzy, regex) and \
            (class_name is None or self.is_match(cls, class_name, fuzzy, regex)):
                matches.append(control)

            try:
                children = control.GetChildren()
            except:
                children = []
            for child in children:
                collect_controls(child, current_depth + 1)

        collect_controls(root)

        if not matches:
            return None

        # 多个匹配，用户选择
        if len(matches) > 1:
            print("\n匹配到多个窗口：")
            for i, m in enumerate(matches):
                try:
                    print(f"[{i + 1}] {m.Name} | 类名: {m.ClassName} | 类型: {m.ControlTypeName}")
                except:
                    print(f"[{i + 1}] 无法读取名称")
            while True:
                choice = input("请选择要激活的窗口编号: ").strip()
                if choice.isdigit() and 1 <= int(choice) <= len(matches):
                    selected = matches[int(choice) - 1]
                    break
                print("⚠️ 输入有误，请重新输入编号。")
        else:
            selected = matches[0]

        # 是否为主窗口
        if selected.ControlTypeName != "Window":
            print("⚠️ 警告：找到的控件不是主窗口，可能是托盘图标。")

        # 激活窗口
        try:
            if not self._is_system_tray_control(selected):
                if hasattr(selected, 'SetTopmost'):
                    selected.SetTopmost(True)
                if hasattr(selected, 'SetActive'):
                    selected.SetActive()
                time.sleep(0.5)
            else:
                print("⚠️ 系统托盘控件不支持激活操作")
        except Exception as e:
            print(f"⚠️ 窗口操作失败: {str(e)}")

        return selected

    def _match_text(self, text: str, pattern: str, fuzzy: bool, regex: bool) -> bool:
        """通用文本匹配方法"""
        if not text or not pattern:
            return False
            
        if regex:
            try:
                return bool(re.search(pattern, text, re.IGNORECASE))
            except:
                return False
        elif fuzzy:
            return pattern.lower() in text.lower()
        else:
            return text.lower() == pattern.lower()
    
    def explore_control_structure(self, control: auto.Control, 
                                  max_depth: int = 20,
                                  keywords: Optional[List[str]] = None,
                                  regex_mode: bool = False) -> Dict[str, Any]:
        """探索控件结构并返回结构化数据"""
        control_info = self._get_control_info(control)
        
        # 关键词过滤
        if keywords and not self._match_keywords(control_info, keywords, regex_mode):
            return None
            
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
        
        if not keywords or self._match_keywords(control_info, keywords, regex_mode):
            self.found_controls.append(control_info)
            return control_info
        return None
    
    def _match_keywords(self, control_info: Dict[str, Any], 
                        keywords: List[str], regex_mode: bool) -> bool:
        """检查控件是否匹配关键词"""
        text_fields = [
            control_info['name'],
            control_info['class_name'],
            control_info['control_type'],
            control_info['automation_id']
        ]
        text_to_search = " ".join(str(field) for field in text_fields if field)
        
        if regex_mode:
            for pattern in keywords:
                try:
                    if re.search(pattern, text_to_search, re.IGNORECASE):
                        return True
                except:
                    continue
            return False
        else:
            text_to_search = text_to_search.lower()
            return any(keyword.lower() in text_to_search for keyword in keywords)
    
    def _get_control_info(self, control: auto.Control) -> Dict[str, Any]:
        """获取控件完整属性信息"""
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
        
        # 基础属性
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
        
        # 坐标信息
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
    
    def print_control_tree(self, control: Dict[str, Any], depth: int = 0, 
                           show_details: bool = False, show_coordinates: bool = False):
        """打印控件树结构"""
        if control is None:
            return
            
        prefix = "  " * depth
        details = []
        
        if show_details:
            if control["class_name"]:
                details.append(f"类名: {control['class_name']}")
            if control["automation_id"]:
                details.append(f"ID: {control['automation_id']}")
            if control["process_id"]:
                details.append(f"进程ID: {control['process_id']}")
        
        if show_coordinates and control["coordinates"]:
            coords = control["coordinates"]
            details.append(f"位置: ({coords['left']}, {coords['top']})")
            details.append(f"大小: {coords['width']}x{coords['height']}")
        
        detail_str = " | " + " | ".join(details) if details else ""
        print(f"{prefix}[{depth}] {control['control_type']} | 名称: '{control['name']}'{detail_str}")
        
        for child in control.get("children", []):
            self.print_control_tree(child, depth + 1, show_details, show_coordinates)
    
    def export_results(self, data: Dict[str, Any], filename: Optional[str] = None,
                       format: OutputFormat = OutputFormat.JSON):
        """导出结果到文件"""
        if not filename:
            desktop = os.path.join(os.path.expanduser("~"), "Desktop")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(desktop, f"ui_scan_{timestamp}.{format.value}")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        if format == OutputFormat.JSON:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        elif format == OutputFormat.CSV:
            self._export_to_csv(data, filename)
        elif format == OutputFormat.TXT:
            self._export_to_txt(data, filename)
        
        print(f"数据已导出到: {os.path.abspath(filename)}")
    
    def _export_to_csv(self, data: Dict[str, Any], filename: str):
        """导出为CSV格式"""
        flattened = []
        self._flatten_controls(data, flattened)
        
        if not flattened:
            return
            
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=flattened[0].keys())
            writer.writeheader()
            writer.writerows(flattened)
    
    def _export_to_txt(self, data: Dict[str, Any], filename: str):
        """导出为TXT格式"""
        with open(filename, 'w', encoding='utf-8') as f:
            self._write_control_text(data, f)
    
    def _write_control_text(self, control: Dict[str, Any], file, depth: int = 0):
        """递归写入控件文本信息"""
        if control is None:
            return
            
        prefix = "  " * depth
        file.write(f"{prefix}[{depth}] {control['control_type']} | 名称: '{control['name']}'\n")
        file.write(f"{prefix}类名: {control.get('class_name', '')}\n")
        file.write(f"{prefix}自动化ID: {control.get('automation_id', '')}\n")
        
        if control.get('coordinates'):
            coords = control['coordinates']
            file.write(f"{prefix}位置: ({coords['left']}, {coords['top']}) 大小: {coords['width']}x{coords['height']}\n")
        
        file.write("\n")
        
        for child in control.get("children", []):
            self._write_control_text(child, file, depth + 1)
    
    def _flatten_controls(self, control: Dict[str, Any], result: List[Dict[str, Any]], 
                          parent_path: str = "", depth: int = 0):
        """扁平化控件结构为CSV行"""
        if not control:
            return
            
        row = {
            "depth": depth,
            "control_path": f"{parent_path}/{control['control_type']}[{control['name']}]",
            **{k: v for k, v in control.items() if k != 'children'}
        }
        
        if control.get('coordinates'):
            row.update({
                'x': control['coordinates']['left'],
                'y': control['coordinates']['top'],
                'width': control['coordinates']['width'],
                'height': control['coordinates']['height']
            })
        
        result.append(row)
        
        for i, child in enumerate(control.get("children", [])):
            self._flatten_controls(child, result, 
                                   f"{row['control_path']}/{i}", 
                                   depth + 1)

def get_user_input():
    """获取用户输入"""
    print("=== 高级GUI结构探查工具 v2.0 ===\n")
    
    # 必填：窗口标题
    while True:
        window_title = input("请输入窗口标题（必须项）: ").strip()
        if window_title:
            break
        print("⚠️  窗口标题是必须的，请重新输入。\n")
    
    # 搜索选项
    class_name = input("请输入窗口类名（可选）: ").strip() or None
    
    print("\n[搜索模式选项]")
    fuzzy = input("使用模糊匹配？(y/n, 默认n): ").strip().lower() == 'y'
    regex = not fuzzy and input("使用正则表达式匹配？(y/n, 默认n): ").strip().lower() == 'y'
    
    # 探查选项
    max_depth = 20
    max_depth_input = input("\n请输入最大递归深度（默认20）: ").strip()
    if max_depth_input.isdigit():
        max_depth = int(max_depth_input)
    
    print("\n[过滤选项]")
    keyword_input = input("输入关键词过滤（多个用逗号分隔，留空不过滤）: ").strip()
    keywords = [k.strip() for k in keyword_input.split(",")] if keyword_input else None
    if keywords:
        regex_mode = input("关键词使用正则表达式匹配？(y/n, 默认n): ").strip().lower() == 'y'
    else:
        regex_mode = False
    
    # 输出选项
    print("\n[输出选项]")
    show_details = input("显示详细信息？(y/n, 默认y): ").strip().lower() != 'n'
    show_coords = input("显示坐标信息？(y/n, 默认y): ").strip().lower() != 'n'
    
    export = input("\n导出结果文件？(y/n, 默认y): ").strip().lower() != 'n'
    if export:
        print("\n[导出选项]")
        print("1. JSON (默认)")
        print("2. CSV")
        print("3. TXT")
        format_choice = input("选择导出格式（1-3, 默认1）: ").strip()
        if format_choice == '2':
            export_format = OutputFormat.CSV
        elif format_choice == '3':
            export_format = OutputFormat.TXT
        else:
            export_format = OutputFormat.JSON
    else:
        export_format = None
    
    return {
        "window_title": window_title,
        "class_name": class_name,
        "fuzzy": fuzzy,
        "regex": regex,
        "max_depth": max_depth,
        "keywords": keywords,
        "regex_mode": regex_mode,
        "show_details": show_details,
        "show_coords": show_coords,
        "export": export,
        "export_format": export_format
    }

def main():
    try:
        params = get_user_input()
        scanner = GUIScanner()
        
        print(f"\n正在查找窗口: '{params['window_title']}'...")
        window = scanner.find_main_window(
            title=params["window_title"],
            class_name=params["class_name"],
            fuzzy=params["fuzzy"],
            regex=params["regex"])
        
        if not window:
            raise Exception(f"未找到匹配窗口。扫描了 {scanner.scan_count} 个控件。")
        
        print(f"\n✅ 找到目标窗口: {window.Name} (类名: {window.ClassName})")
        
        # 新增：激活窗口
        try:
            if not scanner._is_system_tray_control(window):
                if hasattr(window, 'SetTopmost'):
                    window.SetTopmost(True)
                if hasattr(window, 'SetActive'):
                    window.SetActive()
                time.sleep(0.5)
                print("✅ 窗口已激活。")
            else:
                print("⚠️ 目标窗口为系统托盘控件，不支持激活操作。")
        except Exception as e:
            print(f"⚠️ 窗口激活失败: {str(e)}")
        
        print("\n=== 开始分析窗口结构 ===")
        start_time = time.time()
        structure = scanner.explore_control_structure(
            window, 
            max_depth=params["max_depth"],
            keywords=params["keywords"],
            regex_mode=params["regex_mode"])
        elapsed = time.time() - start_time
        
        print(f"\n=== 控件结构 (扫描耗时: {elapsed:.2f}秒, 共 {len(scanner.found_controls)} 个控件) ===\n")
        scanner.print_control_tree(
            structure, 
            show_details=params["show_details"],
            show_coordinates=params["show_coords"])
        
        if params["export"] and structure:
            scanner.export_results(
                structure,
                format=params["export_format"])
        
        print("\n✅ 探查完成！")
    
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
    finally:
        input("\n按回车键退出...")

if __name__ == '__main__':
    main()
