#!/bin/bash
# Usage parse_log.sh caffe.log
# It creates the following two text files, each containing a table:
#     caffe.log.test (columns: '#Iters Seconds TestAccuracy TestLoss')
#     caffe.log.train (columns: '#Iters Seconds TrainingLoss LearningRate')

# 使用方法 parse_log.sh caffe.log
# 这样将会产生两个文件， 每个文件存储为表格的形式
#   caffe.log.test      (Iters Seconds TestAccuracy TestLoss)
#   caffe.log.train     (Iters Seconds TrainingLoss LearningRate)

# get the dirname of the script
# 获取脚本文件的路径
DIR="$( cd "$(dirname "$0")" ; pwd -P )"

# 如果当前参数小于1
if [ "$#" -lt 1 ]
then
echo "Usage parse_log.sh /path/to/your.log"
exit
fi
LOG=`basename $1` # 获取纯粹的文件名，比如 /home/wang/test.txt 将会输出 test.txt
sed -n '/Iteration .* Testing net/,/Iteration *. loss/p' $1 > aux.txt # 将行 "Iteration .* Testing net" 或 "Iteration *. loss" 导入 aut.txt
sed -i '/Waiting for data/d' aux.txt # 删除行 "Waiting for data"
sed -i '/prefetch queue empty/d' aux.txt # 删除行 "prefetch queue empty"
sed -i '/Iteration .* loss/d' aux.txt # 删除行 "Iteration .* loss"
sed -i '/Iteration .* lr/d' aux.txt # 删除行 "Iteration .* lr"
sed -i '/Train net/d' aux.txt # 删除行 "Train net"
# 从 aux.txt 中提取所有 Iteration 的数字， 在 sed 中 \1 代表你前面第一个 \( \) 里面的内容，并将输出到 aux0.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
# 从 aux.txt 中提取所有包含 "Test net output #0" 的行，并用 awk 提取该行的第11个元素，即 accuracy
grep 'Test net output #0' aux.txt | awk '{print $11}' > aux1.txt
# 从 aux.txt 中提取所有包含 "Test net output #1" 的行，并用 awk 提取该行的第11个元素，即 accuracy
grep 'Test net output #1' aux.txt | awk '{print $11}' > aux2.txt

# Extracting elapsed seconds
# For extraction of time since this line contains the start time
# 提取已过去的时间
grep '] Solving ' $1 > aux3.txt
grep 'Testing net' $1 >> aux3.txt
$DIR/extract_seconds.py aux3.txt aux4.txt

# Generating
# 生成 test 文件
echo '#Iters Seconds TestAccuracy TestLoss'> $LOG.test
paste aux0.txt aux4.txt aux1.txt aux2.txt | column -t >> $LOG.test # 合并文件的列
rm aux.txt aux0.txt aux1.txt aux2.txt aux3.txt aux4.txt

# For extraction of time since this line contains the start time
grep '] Solving ' $1 > aux.txt
grep ', loss = ' $1 >> aux.txt
grep 'Iteration ' aux.txt | sed  's/.*Iteration \([[:digit:]]*\).*/\1/g' > aux0.txt
grep ', loss = ' $1 | awk '{print $9}' > aux1.txt
grep ', lr = ' $1 | awk '{print $9}' > aux2.txt

# Extracting elapsed seconds
$DIR/extract_seconds.py aux.txt aux3.txt

# Generating
# 生成 train 文件
echo '#Iters Seconds TrainingLoss LearningRate'> $LOG.train
paste aux0.txt aux3.txt aux1.txt aux2.txt | column -t >> $LOG.train
rm aux.txt aux0.txt aux1.txt aux2.txt  aux3.txt
