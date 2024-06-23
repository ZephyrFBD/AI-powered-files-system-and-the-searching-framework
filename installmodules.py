import subprocess
import pkg_resources
from collections import OrderedDict
from transformers import pipeline
import os

packages = [
    "transformers==4.41.2",
    "tokenizers==0.19.1",
    "torch==2.3.1",
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

# 下面可以继续写你的代码
