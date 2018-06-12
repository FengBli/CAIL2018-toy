# 2018中国“法研杯”法律智能挑战赛 CAIL2018

## 1. Official Website
[2018中国‘法研杯’法律智能挑战赛](http://cail.cipsc.org.cn/index.html)

## 2. Time nodes
- 第一阶段（2018.05.15-2018.07.14）:
    - ~ 6月 5日，基于Small数据的模型提交截至。向评测结果高于基准算法成绩的团队发布Large数据
    - ~ 6月12日，基于Large-test数据对前期模型进行重新评测刷榜
    - ~ 7月14日，最终模型提交截至。
- 第二阶段（2018.07.14-2018.08.14）:
    - 主办方根据一个月的新增数据对最终模型进行封闭评测

## 3. Notice
### 3.1. Necessary adjustment
在将本项目代码clone或download到本地运行时，需要对如下文件处做简单修改：
- 在`./predictor`中创建`model/`目录（github上无法上传空文件夹）
- `./utils/util.py`中的第9行`DATA_DIR`，改为本地数据文件所在目录
- 运行`./test.py`前，将第11行改为测试文件所在目录，第12行改为测试输出结果存放目录
- 运行`./score.py`前，将第187行改为上述测试文件所在目录，第188行改为测试输出结果存放目录

### 3.2. Requirement

- Language Environment
    - Python 3.5

- Packages
    - jieba
    - pandas
    - sklearn

### 3.3. Unfinished Parts
- `./preprocess/*`

## 4. Updates

### 2018-05-18 [feng]
- 数据文件太大，将文件夹从项目中删除
- 默认数据目录为`../data/CAIL2018-small-data`，见`util.py`文件`DATA_DIR`常量
- ~~使用清华中文分词工具[thulac-python](https://github.com/thunlp/THULAC-Python)~~
- thulac分词工具速度过慢，暂时使用jieba，后续可以考虑C++版本的各种分词工具
- **Notice**：法条预测中，有些案件对应多个法条
- 添加`util.py`文件
- 添加`preprocess.py`文件，对数据进行中文分词，整合json2csv文件函数
- 添加`stopwords.txt`文件，来源[GitHub · stopwords-iso/stopwords-zh](https://github.com/stopwords-iso/stopwords-zh)

### 2018-05-26  [feng]
- 使用jieba多线程分词
- 导入从[搜狗词库](https://pinyin.sogou.com/dict/)下载的法律词典
- 删除`CODE_OF_CONDUCT.md`文件
- 添加`dictionary/`文件夹，包含用户词典及由`.scel`(搜狗的用户词典文件)文件解码处理的代码
- 修正`util.py`中的24行的一处bug

### 2018-05-28  [feng]
- 重新组织代码结构，依照官方提供[svm_baseline](https://github.com/thunlp/CAIL2018/tree/master/baseline)代码
- 删除`preprocess.py`
- 添加`train.py`文件, `./predictor/`目录等

### 2018-06-01  [feng]
- 重新组织代码结构：
    - 将`uti.py`,`law.txt`, `accu.txt`, `userdict.txt`等文件均放入`./utils/`目录下
    - 现有的`./predictor/`目录在模型训练完后，即可直接打包提交
    - 添加本地测试与跑分文件：`./test.py`和`./score.py`

## 5. TODOs
- 考虑将停用词处理放入TD-IDF模型内部
- 人工对分词结果进行适当修正
- 对数据进行预分析，即`./preprocess/`目录下相关内容

## 6. Scores

### 0 SVM baseline on small-data
|task-1|task-2|task-3|total-score|
|------|------|------|-----------|
|71.83 |68.79 |47.83 |188.45     |

### 1<sup>st</sup> upload using `linearSVC`

succeeded after 8 stupid attempts by [@FengBlil](https://github.com/FengBli)

date: 05-31

|task-1|task-2|task-3|total-score|
|------|------|------|-----------|
|72.92 |69.43 |52.56 |194.92     |

### 2<sup>nd</sup> upload using `RandomForestClassifier`

date: 06-01

|task-1|task-2|task-3|total-score|
|------|------|------|-----------|
|62.20 |59.99 |48.73 |170.92     |

## 7. Members

Team: 陈-冯-杨

Members:
- 陈子彧 [@mcorange1997](https://github.com/mcorange1997)

- 冯柏淋 [@FengBli](https://github.com/FengBli)

- 杨凌宇 [@Scott1123](https://github.com/Scott1123)
