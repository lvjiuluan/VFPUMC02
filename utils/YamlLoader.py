import os
import yaml

class YamlLoader:
    def __init__(self, base_path=None):
        """
        初始化YamlLoader类，传入根路径。如果没有传入路径，则使用默认路径。
        :param base_path: 要遍历的根路径，默认为 '/root/semi/Semi-supervised-learning/config/classic_cv'
        """
        # 如果没有传入路径，使用默认路径
        if base_path is None:
            self.base_path = "/root/semi/Semi-supervised-learning/config/classic_cv"
        else:
            self.base_path = base_path
        
        self.yaml_data = {}

    def load_yaml_files(self):
        """
        遍历base_path下的所有文件夹，读取指定的yaml文件并保存到字典中
        """
        # 获取base_path下的所有文件夹
        for folder_name in os.listdir(self.base_path):
            folder_path = os.path.join(self.base_path, folder_name)
            
            # 确保是文件夹
            if os.path.isdir(folder_path):
                # 构造yaml文件名
                yaml_file_name = f"{folder_name}_cifar10_250_0.yaml"
                yaml_file_path = os.path.join(folder_path, yaml_file_name)
                
                # 检查yaml文件是否存在
                if os.path.exists(yaml_file_path):
                    # 读取yaml文件并保存到字典中
                    with open(yaml_file_path, 'r') as file:
                        try:
                            yaml_content = yaml.safe_load(file)
                            self.yaml_data[folder_name] = yaml_content
                        except yaml.YAMLError as exc:
                            print(f"Error reading {yaml_file_path}: {exc}")
                else:
                    print(f"YAML file {yaml_file_name} not found in {folder_path}")

    def get_yaml_data(self):
        """
        返回所有已加载的yaml文件内容。
        :return: 包含所有yaml文件内容的字典
        """
        return self.yaml_data


# 使用示例
if __name__ == "__main__":
    # 如果不传入路径，使用默认路径
    loader = YamlLoader()
    loader.load_yaml_files()
    yaml_data = loader.get_yaml_data()

    # 打印加载的yaml数据
    for folder, contents in yaml_data.items():
        print(f"Folder: {folder}")
        for content in contents:
            print(content)
        print("-" * 40)
