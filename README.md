# 2018中国“法研杯”法律智能挑战赛 CAIL2018

## 大赛官方网站
[2018中国‘法研杯’法律智能挑战赛](http://cail.cipsc.org.cn/index.html)

## 时间节点
- 第一阶段（2018.05.15-2018.07.14）:
 - ~ 6月 5日，基于Small数据的模型提交截至。向评测结果高于基准算法成绩的团队发布Large数据
 - ~ 6月12日，基于Large-test数据对前期模型进行重新评测刷榜
 - ~ 7月14日，最终模型提交截至。
- 第二阶段（2018.07.14-2018.08.14）:
 - 主办方根据一个月的新增数据对最终模型进行封闭评测

## Updates

### 2018-05-18
- 数据文件太大，将文件夹从项目中删除
- 默认数据目录为`../data/CAIL2018-small-data`，见`util.py`文件`DATA_DIR`常量
- 使用清华中文分词工具[thulac-python](https://github.com/thunlp/THULAC-Python)
- **Notice**：法条预测中，有些案件对应多个法条
- 添加`util.py`文件
- 添加`preprocess.py`文件，对数据进行中文分词，整合json2csv文件函数
- 添加`stopwords.txt`文件，来源[GitHub·stopwords-iso/stopwords-zh](https://github.com/stopwords-iso/stopwords-zh)

## 团队成员

Team name: 陈-冯-杨

陈子彧 [@mcorange1997](https://github.com/mcorange1997)

冯柏淋 [@FengBli](https://github.com/FengBli)

杨凌宇 [@Scott1123](https://github.com/Scott1123)
