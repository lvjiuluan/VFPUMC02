import logging
import random


class Logger:
    """
    一个静态工具类，用于创建和获取 logger 对象，只输出日志到控制台。
    单例模式
    """
    _logger = None  # 用于存储全局唯一的 logger 实例

    @staticmethod
    def get_logger(log_level=logging.INFO):
        """
        获取全局唯一的 logger 对象。

        :param log_level: 日志级别，默认是 logging.INFO
        :return: logger 对象
        """
        if Logger._logger is None:
            # 创建 logger
            logger = logging.getLogger("ConsoleLogger")
            logger.setLevel(log_level)

            # 防止重复添加 handler
            if not logger.handlers:
                # 创建控制台 handler
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)

                # 定义日志格式
                formatter = logging.Formatter(
                    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                )
                console_handler.setFormatter(formatter)

                # 将 handler 添加到 logger
                logger.addHandler(console_handler)

            # 设置全局唯一的 logger
            Logger._logger = logger

        return Logger._logger

    @staticmethod
    def create_new_logger(log_level=logging.INFO):
        """
        创建一个新的 logger 对象。

        :param log_level: 日志级别，默认是 logging.INFO
        :return: 新的 logger 对象
        """
        # 创建一个新的 logger
        logger = logging.getLogger(f"ConsoleLogger_{random.randint(0, 10000)}")
        logger.setLevel(log_level)

        # 创建控制台 handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # 定义日志格式
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        # 将 handler 添加到 logger
        logger.addHandler(console_handler)

        return logger

    def getLevelName(self, level):
        return logging.getLevelName(level)


logger = Logger.get_logger()
