#! /usr/bin/env python
"""
-*- coding: UTF-8 -*-
Project   : pandapan
Author    : Captain
Email     : qing.ji@extremevision.com.cn
Date      : 2023/11/4 19:27
FileName  : basic_config.py
Software  : PyCharm
Desc      : $END$
"""
from loguru import logger
import os
import langchain

# 是否显示详细日志
log_verbose = True
langchain.verbose = False

# 通常情况下不需要更改以下内容
# 获取项目的根路径
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 创建日志文件的路径
LOG_PATH = os.path.join(root_path, "logs")

# 如果日志文件路径不存在，则创建该路径
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# 添加日志文件的路径到loguru
log_file_path = os.path.join(LOG_PATH, "{time}.log")
logger.add(log_file_path, rotation="500MB",
           encoding="utf-8", retention="10 days")

if __name__ == '__main__':
    logger.info("info日志")
    logger.debug("debug日志")
    logger.warning("warning日志")
    logger.error("error日志")
