def collect_unique_ids():
    """
    不断接收用户输入的id，直到用户输入空行。去重后输出用户输入的id。
    """
    unique_ids = set()  # 使用集合来存储唯一的id
    print("请输入ID（按回车结束）：")
    while True:
        user_input = input().strip()  # 去除前后空格
        if user_input == "":  # 如果输入为空行，结束输入
            break
        unique_ids.add(user_input)  # 将输入的id加入集合，自动去重

    # 输出去重后的id
    print("去重后的ID列表：")
    for uid in unique_ids:
        print(uid)

# 调用函数
collect_unique_ids()
