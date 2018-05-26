#!usr/bin/env python3
# -*- coding: utf-8 -*-


# 数据存放目录
DATA_DIR  = "../data/CAIL2018-small-data"

BR = "\n"

SEP = "\x00"

# print something log info or not
DEBUG = True


def list2str_unicode_version(lst):
    """ NOTICE: maybe **useless**
    将list转换为unicode str，展示对应汉字而非unicode十六进制编码
    可以使用eval(str)得到对应list
    """
    if len(lst) == 1:
        return "[{}]".format(lst[0])
    else:
        return "[{}]".format(", ".join([str(item) for item in lst]))
