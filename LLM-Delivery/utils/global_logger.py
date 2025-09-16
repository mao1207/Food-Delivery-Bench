# -*- coding: utf-8 -*-
"""
全局Logger模块
提供统一的日志记录功能，在整个项目中都可以使用
"""

import logging
import os
from datetime import datetime
from typing import Optional


class GlobalLogger:
    """全局Logger单例类"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not GlobalLogger._initialized:
            self._setup_logger()
            GlobalLogger._initialized = True
    
    def _setup_logger(self, 
                     log_folder: str = "../../log", 
                     file_log_level: int = logging.DEBUG, 
                     console_log_level: int = logging.INFO,
                     log_format: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                     date_format: str = '%Y-%m-%d %H:%M:%S'):
        """
        设置全局logger配置
        
        Args:
            log_folder: 日志文件夹路径
            file_log_level: 文件日志级别
            console_log_level: 控制台日志级别
            log_format: 日志格式
            date_format: 日期格式
        """
        # 确保日志目录存在
        os.makedirs(log_folder, exist_ok=True)
        
        # 创建日志文件名（带时间戳）
        log_file = os.path.join(log_folder, f'delivery_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # 创建根logger
        self.logger = logging.getLogger('delivery_system')
        self.logger.setLevel(logging.DEBUG)
        
        # 清除现有的handlers，避免重复
        self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(log_format, datefmt=date_format)
        
        # 创建文件处理器
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(file_log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 设置其他模块的日志级别
        logging.getLogger('httpx').setLevel(logging.WARNING)
        logging.getLogger('werkzeug').setLevel(logging.ERROR)
        logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    def get_logger(self, name: str) -> logging.Logger:
        """
        获取指定名称的logger
        
        Args:
            name: logger名称
            
        Returns:
            logging.Logger实例
        """
        return self.logger.getChild(name)
    
    def configure(self, 
                 log_folder: Optional[str] = None,
                 file_log_level: Optional[int] = None,
                 console_log_level: Optional[int] = None,
                 log_format: Optional[str] = None,
                 date_format: Optional[str] = None):
        """
        重新配置logger设置
        
        Args:
            log_folder: 日志文件夹路径
            file_log_level: 文件日志级别
            console_log_level: 控制台日志级别
            log_format: 日志格式
            date_format: 日期格式
        """
        # 如果已经初始化，需要重新设置
        if GlobalLogger._initialized:
            self._setup_logger(
                log_folder=log_folder or "log",
                file_log_level=file_log_level or logging.DEBUG,
                console_log_level=console_log_level or logging.INFO,
                log_format=log_format or '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                date_format=date_format or '%Y-%m-%d %H:%M:%S'
            )


# 创建全局logger实例
_global_logger = GlobalLogger()

def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的logger（便捷函数）
    
    Args:
        name: logger名称
        
    Returns:
        logging.Logger实例
    """
    return _global_logger.get_logger(name)

def configure_logger(**kwargs):
    """
    配置全局logger（便捷函数）
    
    Args:
        **kwargs: 配置参数
    """
    _global_logger.configure(**kwargs)

# 导出常用的logger
def get_agent_logger(agent_id: str) -> logging.Logger:
    """获取特定agent的logger"""
    return get_logger(f'agent_{agent_id}')

def get_system_logger() -> logging.Logger:
    """获取系统logger"""
    return get_logger('system')

def get_vlm_logger() -> logging.Logger:
    """获取VLM logger"""
    return get_logger('vlm')
