import sys
from cx_Freeze import setup, Executable

# 定义应用程序的名称、版本等信息
application_name = "VFJReaderTool"
application_version = "1.0"

# 将您的 Python 脚本作为可执行文件的入口点
executables = [Executable('VFJ读写器综合工具.py', base=None, target_name = 'VFJReaderTool')]

# 包含的文件夹及其目标路径（在可执行文件目录下）
include_files = [("DLL", "DLL"),("rapicomm.ini","rapicomm.ini")]

# 定义应用程序的依赖项
includes = ["VFJReader"]
excludes = []
packages = []
build_exe_options = {
    "includes": includes,
    "excludes": excludes,
    "packages": packages
}

# 创建 setup
setup(
    name=application_name,
    version=application_version,
    description="Your Application Description",
    executables=executables,
    options={"build_exe": {
        "includes": includes,
        "excludes": excludes,
        "packages": packages,
        "include_files": include_files
    }},
)
