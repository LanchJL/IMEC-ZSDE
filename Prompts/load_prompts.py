def load_prompts(file_path):
    # 打开文件并逐行读取
    with open(file_path, "r") as file:
        lines = file.readlines()
    # 去除每行末尾的换行符，并存入列表
    text_list = [line.strip() for line in lines]
    # 打印列表内容
    #print("The Prompts are:")
    #print(text_list)
    return text_list