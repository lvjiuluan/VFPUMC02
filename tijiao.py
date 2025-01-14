import subprocess

def run_command(command, success_message, error_message):
    """
    运行系统命令并处理输出和错误。
    :param command: 要运行的命令（列表形式）
    :param success_message: 成功时的提示信息
    :param error_message: 失败时的提示信息
    """
    try:
        subprocess.run(command, check=True)
        print(success_message)
    except subprocess.CalledProcessError as e:
        print(f"{error_message}: {e}")
        exit(1)

def main():
    print("开始执行自动提交操作...")

    # Step 1: git add .
    run_command(
        ["git", "add", "."],
        success_message="已执行: git add .",
        error_message="git add . 执行失败，请检查 Git 仓库状态"
    )

    # Step 2: git commit -m "tijiao"
    run_command(
        ["git", "commit", "-m", "tijiao"],
        success_message="已执行: git commit -m 'tijiao'",
        error_message="git commit 执行失败，请检查是否有需要提交的更改"
    )

    # Step 3: git push
    run_command(
        ["git", "push"],
        success_message="已执行: git push",
        error_message="git push 执行失败，请检查远程仓库连接"
    )

    print("所有操作完成！")

if __name__ == "__main__":
    main()
