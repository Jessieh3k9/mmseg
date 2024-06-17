import subprocess


def run_train_script():
    # 定义命令和参数
    command = "python"
    script = "tools/train.py"
    args = ["frnet_base.py"]

    # 构建完整的命令
    full_command = [command, script] + args

    # 使用 subprocess 运行命令
    result = subprocess.run(full_command)


if __name__ == "__main__":
    run_train_script()
