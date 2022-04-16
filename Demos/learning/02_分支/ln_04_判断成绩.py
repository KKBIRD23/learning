"""
定义两个整数变量 python_score、c_score，判断成绩
只要有一门 >60 分就算合格
"""
python_score = 80
c_score = 50

if python_score > 60 or c_score > 60:
    print("考试通过")
else:
    print("考试失败")