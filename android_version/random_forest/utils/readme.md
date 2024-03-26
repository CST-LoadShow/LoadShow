# 处理数据模块

### dataset.py
重写的两个dataset函数
- MyDataset  处理图片存储的数据集
- CustomTensorDataset  处理tensor存储的数据集 本项目中主要用的

### feature_api.py
计算特征的函数

### get_feature_csv.py
random forest 处理得到csv的函数

**getCsv 输入文件路径得到计算feature之后的csv**
- file_label 选中的应用 需要和文件夹里的文件名符合
- file_path 文件路径
- length 分割原始数据的大小，一般选用64，代表16*64作为一行数据，计算feature
- save_path csv保存路径
- size 指选用原始数据的多少个点，默认就是16个，目前不支持其他大小
- feature_list 选用的特征，目前是17个，不支持选择
- size_max 每个应用最大选用的指纹数量
- return cpu_lists 返回了处理的app的名字，用于后续的处理或者验证

**getCsvNp**
和 getCsv 一样，只是返回的是numpy数组

**getCsvNpSoftware**
输入特定app和指定标签，得到这个应用对应的csv文件
- name  app名字 如 cloudmusic
- label_input  app标签 如 3

**getCsvNpSoftware2**
当一个应用有多个行为，并且分文件夹存储，会把一个应用的不同行为合并起来打成一个标签
- name  app名字 如 cloudmusic
- label_input  app标签 如 3
- return np形式的last_data 和行为列表用于验证

**getCsvChoose**
可以选择只用cpu或者只用gpu的数据

**mergeCSV**
把多个csv文件合并成一个，列需要相同

### get_tensor_dataset.py
处理数据得到tensor的data和label文件

### get_tensor_multi_dataset.py
多应用识别时处理数据

标签有主标签和副标签，根据文件夹名字来判断两个标签


### program_class.py
program_class 映射类别的字典

program_name  映射规范名字的字典
