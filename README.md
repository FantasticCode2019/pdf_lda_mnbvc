# pdf_lda_mnbvc
元数据的lda分类

1.创建一个环境

> conda create -n lda python==3.9
>
> conda activate lda
>
> pip install -r requirements.txt

2.创建分类文件

创建四个分类文件夹0, 1，2，3

> mkdir 0 1 2 3

3.执行脚本

`--source_path`是指元数据的jsonl文件,`--operation`是指将每一行元数据的文件移动或者拷贝到分类文件夹中，参数值有copy，move

> python pdf_class_lda.py --source_path ./data/test.jsonl --operation copy

4.获取最后的`lda_pass4.html`和分类文件夹
