import importlib
import subprocess
import sys

# 安装 setuptools
def install_setuptools():
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools"])

# 尝试导入 pkg_resources
try:
    import pkg_resources
    print("pkg_resources imported successfully.")
    # 打印模块内容
    print(f"pkg_resources module: {pkg_resources}")
except ImportError:
    print("pkg_resources not found. Installing setuptools...")
    install_setuptools()

    # 再次尝试导入 pkg_resources
    try:
        importlib.import_module('pkg_resources')
        import pkg_resources
        print("pkg_resources imported successfully after installation.")
        # 打印模块内容
        print(f"pkg_resources module: {pkg_resources}")
    except ImportError:
        print("Failed to install setuptools. Aborting...")
        sys.exit(1)

import subprocess
import os

packages = [
    "torch==2.3.1",
    "numpy==1.26.4",
    "datasets==2.20.0",
    "gensim==4.3.2",
    "smart-open==7.0.4",
    "scipy==1.12",
    "matplotlib==3.9.0"
]

def check_and_install_packages(packages):
    installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    for package in packages:
        package_name = package.split("==")[0]
        required_version = package.split("==")[1]
        if package_name in installed_packages and installed_packages[package_name] == required_version:
            print(f"{package_name} {required_version} is already installed.")
        else:
            print(f"Installing {package_name} {required_version}...")
            subprocess.run(["pip", "install", "--upgrade", "--force-reinstall", package])

if __name__ == "__main__":
    check_and_install_packages(packages)