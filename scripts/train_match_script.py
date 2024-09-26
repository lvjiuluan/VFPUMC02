from datasets.DataSet import BankDataset
from utils.DataProcessUtils import *
from semilearn import get_config, split_ssl_data, BasicDataset,get_data_loader, get_algorithm, get_net_builder,Trainer
from torchvision import transforms
from enums.HideRatio import HideRatio
import pandas as pd
import logging
from datetime import datetime
from consts.Constants import TOTAL_COUNT

def train_and_get_eval(dataset, config):
    # create model and specify algorithm
    algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)
    data = expand_to_image_shape(normalize_columns(dataset.df.values))
    data = (data * 255).astype(np.uint8)
    target = dataset.y
    lb_data, lb_target, ulb_data, ulb_target = split_ssl_data(config, data, target, 2,
                                                          config.num_labels,ulb_num_labels=config.ulb_num_labels,
                                                          include_lb_to_ulb=config.include_lb_to_ulb)
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                      transforms.RandomCrop(32, padding=int(32 * 0.125), padding_mode='reflect'),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    train_strong_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                 transforms.RandomCrop(32, padding=int(32 * 0.125), padding_mode='reflect'),
                                                 transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    eval_transform = transforms.Compose([transforms.Resize(32),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    lb_dataset = BasicDataset(config.algorithm, lb_data, lb_target, config.num_classes, train_transform, is_ulb=False)
    ulb_dataset = BasicDataset(config.algorithm, lb_data, lb_target, config.num_classes, train_transform, is_ulb=True, strong_transform=train_strong_transform)
    eval_dataset = BasicDataset(config.algorithm, lb_data, lb_target, config.num_classes, eval_transform, is_ulb=False)

    # define data loaders
    train_lb_loader = get_data_loader(config, lb_dataset, config.batch_size)
    train_ulb_loader = get_data_loader(config, ulb_dataset, int(config.batch_size * config.uratio))
    eval_loader = get_data_loader(config, eval_dataset, config.eval_batch_size)

        # training and evaluation
    trainer = Trainer(config, algorithm)
    trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
    return trainer.evaluate(eval_loader)

def run_experiments(algorithms, datasets, yaml_data, initConfig):
    # 创建一个空的字典来存储结果
    results = {}

    # 遍历算法、数据集和隐藏比例
    for algorithm in algorithms:
        baseConfig = yaml_data[algorithm]
        baseConfig.update(initConfig)
        for dataset in datasets:
            for member in HideRatio:
                # 设置配置
                baseConfig['algorithm'] = algorithm
                baseConfig['num_labels'] = nearest_even(TOTAL_COUNT * member.value)
                baseConfig['ulb_num_labels'] = nearest_even(len(dataset.df) * (1 - member.value))

                # 打印日志：记录当前正在运行的算法、数据集和隐藏比例
                logging.info(f"正在运行实验 - 算法: {algorithm}, 数据集: {dataset.baseFileName}, 隐藏比例: {member.name}")
                logging.info(f"配置详情 - 有标签样本数: {baseConfig['num_labels']},无标签样本数: {baseConfig['ulb_num_labels']},一共{baseConfig['num_labels']+ baseConfig['num_labels']}")

                # 获取配置
                config = get_config(baseConfig)

                # 打印日志：记录训练开始时间
                start_time = datetime.now()
                logging.info(f"训练开始 - 算法: {algorithm}, 数据集: {dataset.baseFileName}, 隐藏比例: {member.name}, 开始时间: {start_time}")

                # 训练并获取评估结果
                result = train_and_get_eval(dataset, config)

                # 打印日志：记录训练结束时间
                end_time = datetime.now()
                logging.info(f"训练结束 - 算法: {algorithm}, 数据集: {dataset.baseFileName}, 隐藏比例: {member.name}, 结束时间: {end_time}")
                logging.info(f"训练时长: {end_time - start_time}")

                # 将结果存储在字典中，使用 (algorithm, dataset.baseFileName, member.name) 作为键
                if algorithm not in results:
                    results[algorithm] = {}
                if dataset.baseFileName not in results[algorithm]:
                    results[algorithm][dataset.baseFileName] = {}

                results[algorithm][dataset.baseFileName][member.name] = result

                # 打印日志：记录结果存储的键值信息
                logging.info(f"结果已存储 - 算法: {algorithm}, 数据集: {dataset.baseFileName}, 隐藏比例: {member.name}")

    # 将结果字典转换为 DataFrame
    # 这里使用 pandas 的多级索引功能
    df = pd.DataFrame.from_dict({(alg, ds, mem): res 
                                 for alg, datasets in results.items() 
                                 for ds, members in datasets.items() 
                                 for mem, res in members.items()},
                                orient='index')

    # 设置 DataFrame 的索引名称
    df.index.set_names(['算法', '数据集', '隐藏比例'], inplace=True)

    # 打印日志：记录结果转换为 DataFrame 的操作
    logging.info("结果已成功转换为 DataFrame")

    return df
