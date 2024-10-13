def collect_and_print_ids():
    ids = []  # 用于存储用户输入的ID

    # 不断接收用户输入，直到输入空行
    print("请输入ID（按回车结束输入）：")
    while True:
        user_input = input()
        if user_input == "":  # 如果输入为空行，结束输入
            break
        ids.append(user_input)  # 将输入的ID添加到列表中

    # 按每行9个ID打印，使用制表符分隔
    for i in range(0, len(ids), 9):
        print("\t".join(ids[i:i+9]))

# 调用函数
collect_and_print_ids()
