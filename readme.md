example: python data_analysis_pipeline.py --input_file GroupTwo.csv --clean 1 --process_missing_value mean --process_noise median --transform 1 --convert_data_type 1 --normalize standard_scale,normalize,regularize_L2 --reduce_dims pca --feature_num 45 --output_file result.txt 

处理pipeline：

1. --convert_data_type：是否进行数据类型转换，有str转为float， 取值0、1

2. --clean: 数据清理+过滤：处理缺省值、过滤噪音， 取值0、1，取1时表示进行clean，需选择下面的参数
	（1）--process_missing_value： 处理缺省值，取值可为：del、mean、median
		a）del：直接删除
		b）mean: 均值填充:
		c）median：中值填充
	（2）--del_repeated	：数据去重，取值0、1
	（3）--process_noise：过滤噪音，取值mean、median
		a）mean：均值过滤器
		b）median：中值过滤器
		

3. --transform：数据转换，取值0、1，取1时表示进行数据转换，需选择下面的参数
	（1）--normalize：数据转换，取值为：standard_scale,normalize,regularize_L1,regularize_L2， 可选多个参数，用,分隔
		a）standard_scale：数据转换为0均值0方差数据
		b）normalize：数据归一化
		c）regularize_L1：数据进行L1正则化
		d）regularize_L2：数据进行L2正则化


4. --reduce_dims：数据降维， 取值为pca或者svd， 用pca或者svd降维
	（1）feature_num：降维时指定选取的top feature的数目

