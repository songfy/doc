
<h1>目录<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#pandas基本概念" data-toc-modified-id="pandas基本概念-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>pandas基本概念</a></span><ul class="toc-item"><li><span><a href="#pandas数据结构剖析" data-toc-modified-id="pandas数据结构剖析-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>pandas数据结构剖析</a></span><ul class="toc-item"><li><span><a href="#Series" data-toc-modified-id="Series-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Series</a></span></li><li><span><a href="#DataFrame" data-toc-modified-id="DataFrame-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>DataFrame</a></span></li><li><span><a href="#索引" data-toc-modified-id="索引-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>索引</a></span></li><li><span><a href="#pandas基本操作" data-toc-modified-id="pandas基本操作-1.1.4"><span class="toc-item-num">1.1.4&nbsp;&nbsp;</span>pandas基本操作</a></span><ul class="toc-item"><li><span><a href="#重索引" data-toc-modified-id="重索引-1.1.4.1"><span class="toc-item-num">1.1.4.1&nbsp;&nbsp;</span>重索引</a></span></li><li><span><a href="#丢弃一行或者一列" data-toc-modified-id="丢弃一行或者一列-1.1.4.2"><span class="toc-item-num">1.1.4.2&nbsp;&nbsp;</span>丢弃一行或者一列</a></span></li><li><span><a href="#数据选取" data-toc-modified-id="数据选取-1.1.4.3"><span class="toc-item-num">1.1.4.3&nbsp;&nbsp;</span>数据选取</a></span></li><li><span><a href="#数据对齐" data-toc-modified-id="数据对齐-1.1.4.4"><span class="toc-item-num">1.1.4.4&nbsp;&nbsp;</span>数据对齐</a></span></li></ul></li><li><span><a href="#pandas函数简单介绍" data-toc-modified-id="pandas函数简单介绍-1.1.5"><span class="toc-item-num">1.1.5&nbsp;&nbsp;</span>pandas函数简单介绍</a></span><ul class="toc-item"><li><span><a href="#apply和applymap函数" data-toc-modified-id="apply和applymap函数-1.1.5.1"><span class="toc-item-num">1.1.5.1&nbsp;&nbsp;</span>apply和applymap函数</a></span></li><li><span><a href="#排序函数" data-toc-modified-id="排序函数-1.1.5.2"><span class="toc-item-num">1.1.5.2&nbsp;&nbsp;</span>排序函数</a></span></li><li><span><a href="#汇总计算函数" data-toc-modified-id="汇总计算函数-1.1.5.3"><span class="toc-item-num">1.1.5.3&nbsp;&nbsp;</span>汇总计算函数</a></span></li></ul></li><li><span><a href="#缺失值的处理" data-toc-modified-id="缺失值的处理-1.1.6"><span class="toc-item-num">1.1.6&nbsp;&nbsp;</span>缺失值的处理</a></span></li><li><span><a href="#整数索引" data-toc-modified-id="整数索引-1.1.7"><span class="toc-item-num">1.1.7&nbsp;&nbsp;</span>整数索引</a></span></li></ul></li></ul></li></ul></div>

## pandas基本概念

### pandas数据结构剖析

#### Series

**Series = 一组数据 + 数据标签**
     
如果从表格的角度来看, 就是表格里面的一列带着索引.


```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

obj = Series(data=[1, 2, 3, 4])
print(obj)

# 我们可以用values和index分别获取数据和标签
print(obj.values)
print(type(obj.values))
print(obj.index)
print(type(obj.index))
```

    0    1
    1    2
    2    3
    3    4
    dtype: int64
    [1 2 3 4]
    <class 'numpy.ndarray'>
    RangeIndex(start=0, stop=4, step=1)
    <class 'pandas.core.indexes.range.RangeIndex'>
    

   我们发现, 这个obj.values其实就是一个numpy对象, 而obj.index是一个index对象, 我们也可以在代码指定index:


```python
obj2 = Series(data=[1, 2, 3, 4], index=['a', 'b', 'c', 'd'], dtype=np.float32, name='test')
print(obj2)
```

    a    1.0
    b    2.0
    c    3.0
    d    4.0
    Name: test, dtype: float32
    

 我们要注意, **Series的标签和数据的对应关系, 不随运算而改变**. 大部分在np里面可以使用的操作, 在Series上都是行得通的.


```python
print(obj2 * 2)
print(obj2 ** 2)
print(np.exp(obj2))
```

    a    2.0
    b    4.0
    c    6.0
    d    8.0
    Name: test, dtype: float32
    a     1.0
    b     4.0
    c     9.0
    d    16.0
    Name: test, dtype: float32
    a     2.718282
    b     7.389056
    c    20.085537
    d    54.598148
    Name: test, dtype: float32
    

Series其实也可以看作是一个字典, 我们来看这么个例子:


```python
obj3 = Series({'a':1, 'b':2})
print(obj3)
print('a' in obj3)
```

    a    1
    b    2
    dtype: int64
    True
    

我们照样可以改变索引:


```python
obj3 = Series({'a':1, 'b':2}, index=['a', 'c'])
print(obj3)
```

    a    1.0
    c    NaN
    dtype: float64
    

NaN表示缺失数据, 我们可以用isnull和notnull看来判断:


```python
print(pd.isnull(obj3))
print(pd.notnull(obj3))
```

    a    False
    c     True
    dtype: bool
    a     True
    c    False
    dtype: bool
    

我们来看一个加法的例子:


```python
a = Series([1, 2], index=['a', 'b'])
b = Series([2, 3], index=['b', 'c'])
print(a + b)
```

    a    NaN
    b    4.0
    c    NaN
    dtype: float64
    

**两个Series对象在做运算时, 会自动对齐**.

#### DataFrame

DataFrame可以看作是一张表, 我们也可以把它看作是横向和纵向两个Series.

构造DataFrame可以通过一个ndarray或者是list的字典来构造.


```python
data = {'a':[1, 2, 3], 'b':[4, 5, 6] }
frame = DataFrame(data)
print(frame)
```

       a  b
    0  1  4
    1  2  5
    2  3  6
    

我们一样也可以指定index, 不过对于DataFrame, 要指定columns和index.


```python
frame = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])
print(frame)
```

       a  b    c
    A  1  4  NaN
    B  2  5  NaN
    C  3  6  NaN
    

我们可以用一个二维ndarray来初始化DataFrame:


```python
data = np.random.randn(3, 3)
frame = DataFrame(data)
print(frame)

print("-----------------------------------------------------------------")

frame = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])
print(frame)
```

              0         1         2
    0 -0.219102 -1.849864 -0.545050
    1 -2.639734  0.489300  0.911258
    2  0.810350  0.383256 -0.815600
    -----------------------------------------------------------------
              a         b         c
    A -0.219102 -1.849864 -0.545050
    B -2.639734  0.489300  0.911258
    C  0.810350  0.383256 -0.815600
    

我们用索引的方式, 可以获取到某一列, 作为Series.


```python
print(frame['a'])
```

    A   -0.219102
    B   -2.639734
    C    0.810350
    Name: a, dtype: float64
    

我们发现, 这个Series的名字, 就是列名.

如果我们要修改DataFrame的某个值, 怎么改呢? 我们先来看看怎么修改某一列的值.


```python
# 修改某一个列的值
frame['a'] = [1,1,1]
print(frame)
```

       a         b         c
    A  1 -1.849864 -0.545050
    B  1  0.489300  0.911258
    C  1  0.383256 -0.815600
    


```python
# 如果我们list的长度不符合, 就会报错
frame['a'] = [1,1]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-81-54000a6faf10> in <module>()
          1 # 如果我们list的长度不符合, 就会报错
    ----> 2 frame['a'] = [1,1]
    

    d:\anaconda3\envs\tensorflow_gpu\lib\site-packages\pandas\core\frame.py in __setitem__(self, key, value)
       2329         else:
       2330             # set column
    -> 2331             self._set_item(key, value)
       2332 
       2333     def _setitem_slice(self, key, value):
    

    d:\anaconda3\envs\tensorflow_gpu\lib\site-packages\pandas\core\frame.py in _set_item(self, key, value)
       2395 
       2396         self._ensure_valid_index(value)
    -> 2397         value = self._sanitize_column(key, value)
       2398         NDFrame._set_item(self, key, value)
       2399 
    

    d:\anaconda3\envs\tensorflow_gpu\lib\site-packages\pandas\core\frame.py in _sanitize_column(self, key, value, broadcast)
       2566 
       2567             # turn me into an ndarray
    -> 2568             value = _sanitize_index(value, self.index, copy=False)
       2569             if not isinstance(value, (np.ndarray, Index)):
       2570                 if isinstance(value, list) and len(value) > 0:
    

    d:\anaconda3\envs\tensorflow_gpu\lib\site-packages\pandas\core\series.py in _sanitize_index(data, index, copy)
       2877 
       2878     if len(data) != len(index):
    -> 2879         raise ValueError('Length of values does not match length of ' 'index')
       2880 
       2881     if isinstance(data, PeriodIndex):
    

    ValueError: Length of values does not match length of index



```python
# 对于这种情况, 我们可以用Series来搞定
s = Series({'A':1, 'B':1})
frame['a'] = s
print(frame)
```

         a         b         c
    A  1.0 -1.849864 -0.545050
    B  1.0  0.489300  0.911258
    C  NaN  0.383256 -0.815600
    

如果我们要新加一列, 可以这样写:


```python
frame['d'] = s
print(frame)
```

         a         b         c    d
    A  1.0 -1.849864 -0.545050  1.0
    B  1.0  0.489300  0.911258  1.0
    C  NaN  0.383256 -0.815600  NaN
    

如果我们要删除一列, 可以用del


```python
print(frame.columns)
del frame['d']
print(frame.columns)
```

    Index(['a', 'b', 'c', 'd'], dtype='object')
    Index(['a', 'b', 'c'], dtype='object')
    

我们还可以用字典来初始化我们的DataFrame:


```python
data = {'a':{'A':1,'B':2}, 'b':{'A':3,'B':4}}
frame = DataFrame(data)
print(frame)
print(frame.columns)
print(frame.index)
print(frame.values)
```

       a  b
    A  1  3
    B  2  4
    Index(['a', 'b'], dtype='object')
    Index(['A', 'B'], dtype='object')
    [[1 3]
     [2 4]]
    

有Series的字典也可以初始化DataFrame, 我们这里不演示了, 留作作业.

**作业1: 用Series的字典初始化DataFrame.**

#### 索引

下面我们来里聊聊索引, 这个也是pandas中比较基本的一个概念.


```python
index = frame.index
colunms = frame.columns
print(type(colunms))
print(type(index))
print(index.name)
print(colunms.name)
```

    <class 'pandas.core.indexes.base.Index'>
    <class 'pandas.core.indexes.base.Index'>
    None
    None
    

我们可以把索引理解为 ** 不可变的hashset **


```python
print(index)
```

    Index(['A', 'B'], dtype='object')
    


```python
index[1]='D'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-88-e57751cb899e> in <module>()
    ----> 1 index[1]='D'
    

    d:\anaconda3\envs\tensorflow_gpu\lib\site-packages\pandas\core\indexes\base.py in __setitem__(self, key, value)
       1668 
       1669     def __setitem__(self, key, value):
    -> 1670         raise TypeError("Index does not support mutable operations")
       1671 
       1672     def __getitem__(self, key):
    

    TypeError: Index does not support mutable operations


注意这行错误:
     Index does not support mutable operations


此外, 索引也支持hashset类似的操作, 我们这里不提了.

**作业2: 熟悉index的操作函数.**

#### pandas基本操作

我们这里来看看pands具体有哪些基本操作.

##### 重索引

在已经有的索引的基础上改变索引的操作我们称为重索引.


```python
data = np.random.randn(3, 3)
frame = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])
print(frame)
```

              a         b         c
    A -0.545762 -1.089096 -2.164863
    B -0.514410 -1.337000  0.434980
    C -1.729591 -1.060305 -0.218492
    


```python
# 将A, B, C重索引C, B, A
frame = frame.reindex(index=['C', 'B', 'A'])
print(frame)
```

              a         b         c
    C -1.729591 -1.060305 -0.218492
    B -0.514410 -1.337000  0.434980
    A -0.545762 -1.089096 -2.164863
    


```python
# 同时将列也索引成c, b, a
frame = frame.reindex(index=['C', 'B', 'A'], columns=['c', 'b', 'a'])
print(frame)
```

              c         b         a
    C -0.218492 -1.060305 -1.729591
    B  0.434980 -1.337000 -0.514410
    A -2.164863 -1.089096 -0.545762
    


```python
# 加入d 新的索引
frame = frame.reindex(index=['C', 'B', 'A'], columns=['c', 'b', 'a', 'd'])
print(frame)
```

              c         b         a   d
    C -0.218492 -1.060305 -1.729591 NaN
    B  0.434980 -1.337000 -0.514410 NaN
    A -2.164863 -1.089096 -0.545762 NaN
    


```python
# 将NaN值设置为0
frame = frame.reindex(index=['C', 'B', 'A', 'D'], columns=['c', 'b', 'a', 'd'], fill_value=0)
print(frame)
```

              c         b         a    d
    C -0.218492 -1.060305 -1.729591  NaN
    B  0.434980 -1.337000 -0.514410  NaN
    A -2.164863 -1.089096 -0.545762  NaN
    D  0.000000  0.000000  0.000000  0.0
    

**注意到, 因为d列已经是NaN里, 所以不会填充, 填充只发生在reindex的时候.**

##### 丢弃一行或者一列

我们可以很方便的丢弃一行或者一列:



```python
frame = frame.drop(['C', 'B'])
print(frame)
```

              c         b         a    d
    A -2.164863 -1.089096 -0.545762  NaN
    D  0.000000  0.000000  0.000000  0.0
    


```python
# 丢弃一列
frame = frame.drop([ 'd'], axis=1)
print(frame)
```

              c         b         a
    A -2.164863 -1.089096 -0.545762
    D  0.000000  0.000000  0.000000
    

##### 数据选取

我们用[]一般选取的是一列, 当然也有几种特殊的情况:


```python
data = np.random.randn(3, 3)
frame = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])
print(frame[:2])
print('-----------------------------------------------------------------')
print(frame[1:2])
```

              a         b         c
    A  0.542189 -2.005171  1.265211
    B -0.145857 -0.054910 -1.277347
    -----------------------------------------------------------------
              a        b         c
    B -0.145857 -0.05491 -1.277347
    

frame[:2]并不是大家理解中的倒数两列, 而是倒数两行. 也是奇葩, 作者真是脑子有坑.


```python
print(frame > 0.1)
```

           a      b      c
    A   True  False   True
    B  False  False  False
    C   True   True   True
    


```python
frame[frame > 0.1] = 0
print(frame)
```

              a         b         c
    A  0.000000 -2.005171  0.000000
    B -0.145857 -0.054910 -1.277347
    C  0.000000  0.000000  0.000000
    

如果我们要选取特定一行呢, 我们会用到loc和ix函数.


```python
print(frame.loc['A'])
print("-----------------------------------------------------")
print(frame.loc['A':'B'])
print("-----------------------------------------------------")
print(frame.loc['A', 'c'])
print("-----------------------------------------------------")
print(frame.loc['A':'B', 'b':'c'])
print("-----------------------------------------------------")
print(frame.ix[:2, :2])
print("-----------------------------------------------------")
print(frame.ix['A':'B', 'b':'c'])
print("-----------------------------------------------------")
print(frame.ix['B', 'b':'c'])
```

    a    0.000000
    b   -2.005171
    c    0.000000
    Name: A, dtype: float64
    -----------------------------------------------------
              a         b         c
    A  0.000000 -2.005171  0.000000
    B -0.145857 -0.054910 -1.277347
    -----------------------------------------------------
    0.0
    -----------------------------------------------------
              b         c
    A -2.005171  0.000000
    B -0.054910 -1.277347
    -----------------------------------------------------
              a         b
    A  0.000000 -2.005171
    B -0.145857 -0.054910
    -----------------------------------------------------
              b         c
    A -2.005171  0.000000
    B -0.054910 -1.277347
    -----------------------------------------------------
    b   -0.054910
    c   -1.277347
    Name: B, dtype: float64
    

** 注意: 对于tag, 我们两端点是可以取到的. **

##### 数据对齐

我们会来讲讲数据对齐的问题, Series的对齐我们看过了, DataFrame会在行和列两个维度上进行对齐.


```python
data = np.random.randn(3, 3)

a = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])
b = DataFrame(data, index=['A', 'B', 'C'], columns=['d', 'c', 'b'])

print(a)
print("-----------------------------------------------------------------")
print(b)
print("-----------------------------------------------------------------")
print(a + b)
```

              a         b         c
    A  0.148959 -1.091292 -1.371497
    B  0.073900  0.339610  0.236825
    C  1.130694  0.069362  0.534634
    -----------------------------------------------------------------
              d         c         b
    A  0.148959 -1.091292 -1.371497
    B  0.073900  0.339610  0.236825
    C  1.130694  0.069362  0.534634
    -----------------------------------------------------------------
        a         b         c   d
    A NaN -2.462789 -2.462789 NaN
    B NaN  0.576435  0.576435 NaN
    C NaN  0.603996  0.603996 NaN
    


```python
# 设置缺失值
print(a.add(b, fill_value=0))
```

              a         b         c         d
    A  0.148959 -2.462789 -2.462789  0.148959
    B  0.073900  0.576435  0.576435  0.073900
    C  1.130694  0.603996  0.603996  1.130694
    

DataFrame和series相加, 这里会用到numpy里面提到过的广播机制. 大家可以去numpy里面看看:


```python
a = a.add(b, fill_value=0)
print(a)
print("-------------------------------------------------------")
b = a.iloc [0]
print(b)
```

              a         b         c         d
    A  0.148959 -2.462789 -2.462789  0.148959
    B  0.073900  0.576435  0.576435  0.073900
    C  1.130694  0.603996  0.603996  1.130694
    -------------------------------------------------------
    a    0.148959
    b   -2.462789
    c   -2.462789
    d    0.148959
    Name: A, dtype: float64
    

我们注意到, a的大小是(3, 4), b的大小是(1,4), tail dimenson是一致的, 可以在其他维度上进行广播.


```python
print(a - b)
```

              a         b         c         d
    A  0.000000  0.000000  0.000000  0.000000
    B -0.075059  3.039224  3.039224 -0.075059
    C  0.981735  3.066785  3.066785  0.981735
    

如果我们要在列上广播呢, 我们来看下面这个例子:
    


```python
b = a['b']
print(b)
print("-------------------------------------------------------")
print(a - b) 
```

    A   -2.462789
    B    0.576435
    C    0.603996
    Name: b, dtype: float64
    -------------------------------------------------------
        A   B   C   a   b   c   d
    A NaN NaN NaN NaN NaN NaN NaN
    B NaN NaN NaN NaN NaN NaN NaN
    C NaN NaN NaN NaN NaN NaN NaN
    

我们发现, 默认是在axis=1(**横向**)做减法的, 因为用了广播机制, 同时缺失值为NaN, 我们就得到了上述的结果, 但是这并不是我们想要的.

我们可以设置需要广播以及运算的轴


```python
print(a.sub(b, axis=0))
```

              a    b    c         d
    A  2.611748  0.0  0.0  2.611748
    B -0.502535  0.0  0.0 -0.502535
    C  0.526698  0.0  0.0  0.526698
    

 axis=0表示**纵向**减法.

#### pandas函数简单介绍

这里我们会简单介绍一些pandas的一些函数, 我们来看看, DataFrame上可以定义哪些有趣的函数.

##### apply和applymap函数

首先, 我们来介绍一下apply函数, 这个函数可以把我们自定义的函数用在DataFrame的每一行或者每一列上, 我们来看一个例子.


```python
data = np.random.randn(3, 3)

a = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])

print(a)

print("------------------------------------------------------------------------")

print(a.apply(np.max))
print(a.apply(np.min))
print(a.apply(lambda x: np.max(x) - np.min(x)))

print("------------------------------------------------------------------------")
a.loc['max'] = a.apply(np.max)
a.loc['min'] = a.apply(np.min)
a.loc['margin'] = a.apply(lambda x: np.max(x) - np.min(x))
print(a)
```

              a         b         c
    A -0.137231 -2.008153 -0.766172
    B  0.473756 -0.071421  1.549933
    C  1.057641  0.375983 -0.647798
    ------------------------------------------------------------------------
    a    1.057641
    b    0.375983
    c    1.549933
    dtype: float64
    a   -0.137231
    b   -2.008153
    c   -0.766172
    dtype: float64
    a    1.194872
    b    2.384135
    c    2.316105
    dtype: float64
    ------------------------------------------------------------------------
                   a         b         c
    A      -0.137231 -2.008153 -0.766172
    B       0.473756 -0.071421  1.549933
    C       1.057641  0.375983 -0.647798
    max     1.057641  0.375983  1.549933
    min    -0.137231 -2.008153 -0.766172
    margin  1.194872  2.384135  2.316105
    


```python

a['max'] = a.apply(np.max, axis=1)
a['min'] = a.apply(np.min, axis=1)
a['margin'] = a.apply(lambda x: np.max(x) - np.min(x), axis=1)
print(a)
```

                   a         b         c       max       min    margin
    A      -0.137231 -2.008153 -0.766172 -0.137231 -2.008153  1.870922
    B       0.473756 -0.071421  1.549933  1.549933 -0.071421  1.621354
    C       1.057641  0.375983 -0.647798  1.057641 -0.647798  1.705439
    max     1.057641  0.375983  1.549933  1.549933  0.375983  1.173950
    min    -0.137231 -2.008153 -0.766172 -0.137231 -2.008153  1.870922
    margin  1.194872  2.384135  2.316105  2.384135  1.194872  1.189263
    

**我们对轴要牢记, 0是在每一列上做, 1是在每一行上做.** 这个例子希望大家好好理解.

我们如果想元素级的调用我们的函数, 我们可以用函数applymap函数:


```python
print(a)
```

                   a         b         c       max       min    margin
    A      -0.137231 -2.008153 -0.766172 -0.137231 -2.008153  1.870922
    B       0.473756 -0.071421  1.549933  1.549933 -0.071421  1.621354
    C       1.057641  0.375983 -0.647798  1.057641 -0.647798  1.705439
    max     1.057641  0.375983  1.549933  1.549933  0.375983  1.173950
    min    -0.137231 -2.008153 -0.766172 -0.137231 -2.008153  1.870922
    margin  1.194872  2.384135  2.316105  2.384135  1.194872  1.189263
    

如果我们想要把小数保留两位, 怎么搞呢, 我们可以看看在python中怎么保留两位小数的


```python
x = 0.123456
print('%.2f' %  x)
```

    0.12
    

我们可以这么来


```python
print(a)

b = a.copy()


b = b.applymap(lambda x: float('%.2f' %  x))
print("---------------------------------------------------------------------------------------------------------")
print(b)
```

                   a         b         c       max       min    margin
    A      -0.137231 -2.008153 -0.766172 -0.137231 -2.008153  1.870922
    B       0.473756 -0.071421  1.549933  1.549933 -0.071421  1.621354
    C       1.057641  0.375983 -0.647798  1.057641 -0.647798  1.705439
    max     1.057641  0.375983  1.549933  1.549933  0.375983  1.173950
    min    -0.137231 -2.008153 -0.766172 -0.137231 -2.008153  1.870922
    margin  1.194872  2.384135  2.316105  2.384135  1.194872  1.189263
    ---------------------------------------------------------------------------------------------------------
               a     b     c   max   min  margin
    A      -0.14 -2.01 -0.77 -0.14 -2.01    1.87
    B       0.47 -0.07  1.55  1.55 -0.07    1.62
    C       1.06  0.38 -0.65  1.06 -0.65    1.71
    max     1.06  0.38  1.55  1.55  0.38    1.17
    min    -0.14 -2.01 -0.77 -0.14 -2.01    1.87
    margin  1.19  2.38  2.32  2.38  1.19    1.19
    

** 作业3: 对整个表格的浮点数进行四舍五入. **

##### 排序函数

排序一直是一个很重要的课题, 


```python
a = Series(range(4), index=['a', 'c', 'b', 'd'])
print(a)
print("--------------------------------")
print(a.sort_index())
print("--------------------------------")
print(a.sort_index(ascending=False))
```

    a    0
    c    1
    b    2
    d    3
    dtype: int32
    --------------------------------
    a    0
    b    2
    c    1
    d    3
    dtype: int32
    --------------------------------
    d    3
    c    1
    b    2
    a    0
    dtype: int32
    

我们可以发现, sort_index这个函数在Series上是按照index来排序的.

对于DataFrame来说, 我们可以按照某一列来排序, 用sort_values函数.


```python
data = np.random.randn(3, 3)

a = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])

print(a)

print('--------------------------------------------------------------------')

print(a.sort_values(by='b'))
```

              a         b         c
    A -0.247148  0.117866  0.485004
    B -1.303519 -1.073757 -1.265169
    C  0.647247  1.171883 -0.301084
    --------------------------------------------------------------------
              a         b         c
    B -1.303519 -1.073757 -1.265169
    A -0.247148  0.117866  0.485004
    C  0.647247  1.171883 -0.301084
    

另外, 我们要注意的是, 这个索引是可以用重复的, 我们可以来看看:


```python
data = np.random.randn(3, 3)

a = DataFrame(data, index=['A', 'B', 'B'], columns=['a', 'b', 'c'])

print(a.loc['B'])
```

              a         b         c
    B -0.364841  0.765296  1.869450
    B  0.624467  0.929586  0.673822
    

##### 汇总计算函数

把一串数字经过计算变成一个数字, 这样的操作, 我们称之为汇总. 我们有很多的函数可以帮助我们完成汇总的操作,

比如mean, sum, media等等, 下面我们来看看这些例子.


```python
data = {'a': list(range(1, 4)), 'b': list(range(1, 4)), 'c': list(range(1, 4))}

frame = DataFrame(data, index=['A', 'B', 'C'])

print(frame)

print('--------------------------------------------------------------------------')

# 把每一行加起来求和
print(frame.sum())

print('--------------------------------------------------------------------------')

# 把每一列加起来求和
print(frame.sum(axis=1))

print('--------------------------------------------------------------------------')

frame['sum'] = frame.sum(axis=1)
frame.loc['sum'] = frame.sum()
print(frame)

print('--------------------------------------------------------------------------')

frame = DataFrame(data, index=['A', 'B', 'C'])
frame['sum'] = frame.median(axis=1)
frame.loc['sum'] = frame.median()
print(frame)
```

       a  b  c
    A  1  1  1
    B  2  2  2
    C  3  3  3
    --------------------------------------------------------------------------
    a    6
    b    6
    c    6
    dtype: int64
    --------------------------------------------------------------------------
    A    3
    B    6
    C    9
    dtype: int64
    --------------------------------------------------------------------------
         a  b  c  sum
    A    1  1  1    3
    B    2  2  2    6
    C    3  3  3    9
    sum  6  6  6   18
    --------------------------------------------------------------------------
           a    b    c  sum
    A    1.0  1.0  1.0  1.0
    B    2.0  2.0  2.0  2.0
    C    3.0  3.0  3.0  3.0
    sum  2.0  2.0  2.0  2.0
    

在这里我们要单独讲一讲这个NaN的值的处理, NaN可以选择跳过, 也可以不跳过, 我们来看看:


```python
data = {'a': list(range(1, 4)), 'b': list(range(1, 4)), 'c': list(range(1, 4))}

frame = DataFrame(data, index=['A', 'B', 'C'])

frame['a']['A'] = np.nan

print(frame)

print(frame.sum())
```

         a  b  c
    A  NaN  1  1
    B  2.0  2  2
    C  3.0  3  3
    a    5.0
    b    6.0
    c    6.0
    dtype: float64
    

我们发现, 这个NaN被略过去了, 我们也可以选择不略过去.


```python
print(frame.sum(skipna=False))
```

    a    NaN
    b    6.0
    c    6.0
    dtype: float64
    

我们这里要提两个数学概念, 协方差和相关系数, 我们来看看这两个的定义:

协方差:

$$ Cov(X, Y) = E[(X - E[X])(Y - E[Y])] = \frac{1}{n^2}\sum_{i=1, j=1}^n(X_{i}Y_{j}-\bar{X}\bar{Y})$$

这是我们协方差的定义, 我们可以看看相关系数的定义:

$$ \rho = \frac{ Cov(X, Y)}{\sigma_{X}\sigma_{Y}} $$

这样, 我们就把这两个概念用公式来表达出来了, 我们来看看具体怎么计算.


```python
data = np.random.randn(3, 3)

a = DataFrame(data, index=['A', 'B', 'B'], columns=['a', 'b', 'c'])

print(a.corr())

print(a.cov())
```

              a         b         c
    a  1.000000 -0.804968 -0.630290
    b -0.804968  1.000000  0.046734
    c -0.630290  0.046734  1.000000
              a         b         c
    a  0.281901 -0.189824 -0.155039
    b -0.189824  0.197264  0.009616
    c -0.155039  0.009616  0.214637
    

** 作业4: 了解一下相关系数和协方差的数学意义.https://www.zhihu.com/question/20852004 **

另外, 我们还有一个非常实用的函数, pd.value_counts


```python
data = {'a': list(range(1, 4)), 'b': list(range(1, 4)), 'c': list(range(1, 4))}

frame = DataFrame(data, index=['A', 'B', 'C'])

print(frame.apply(pd.value_counts))
```

       a  b  c
    3  1  1  1
    2  1  1  1
    1  1  1  1
    

a, b c三列中, 1, 2, 3各出现1次.


```python
print(frame.apply(pd.value_counts, axis=1).fillna(0))
```

         1    2    3
    A  3.0  0.0  0.0
    B  0.0  3.0  0.0
    C  0.0  0.0  3.0
    

A行1出现3次, B行2出现3次, C行3出现3次

#### 缺失值的处理

我们这里要谈一个非常重要的概念, 缺失值的处理, 大致上看, 关于缺失值, 可以用丢弃和填充两种方法.

因为我们拿到的数据, 不可能所有都是完整的, 所以我们需要对这些缺失值进行处理

首先, 我们来看看怎么判断有缺失值, 我们可以用isnull来判断:


```python
data = np.random.randn(3, 3)

a = DataFrame(data, index=['A', 'B', 'C'], columns=['a', 'b', 'c'])

a['a']['A'] = None

a['b']['C'] = None

print(a)
```

              a         b         c
    A       NaN -1.577756  0.572846
    B -1.045493 -0.852665 -0.374464
    C -1.419523       NaN  0.267812
    


```python
print(a.isnull())
```

           a      b      c
    A   True  False  False
    B  False  False  False
    C  False   True  False
    

进一步我们看看每一列都有多少NaN值.


```python
print(a.isnull().sum())
print(a.isnull().sum(axis=1))
```

    a    1
    b    1
    c    0
    dtype: int64
    A    1
    B    0
    C    1
    dtype: int64
    

我们可以过滤掉所有的有NaN的行或者列, 我们来看看:


```python
print(a.dropna())

print("-------------------------------------")

print(a.dropna(axis=1))

print("-------------------------------------")

print(a.dropna(how='all'))
```

              a         b         c
    B -1.045493 -0.852665 -0.374464
    -------------------------------------
              c
    A  0.572846
    B -0.374464
    C  0.267812
    -------------------------------------
              a         b         c
    A       NaN -1.577756  0.572846
    B -1.045493 -0.852665 -0.374464
    C -1.419523       NaN  0.267812
    

我们来看看怎么对于缺失值来进行填充, fillna我们在前面已经看过一个具体的例子了, 我们来看看别的例子:


```python
print(a.fillna({'a':0, 'b':1})) # 这是用不同值对不同列进行na的填充

print(a.fillna(Series({'a':0, 'b':1})))
```

              a         b         c
    A  0.000000 -1.577756  0.572846
    B -1.045493 -0.852665 -0.374464
    C -1.419523  1.000000  0.267812
              a         b         c
    A  0.000000 -1.577756  0.572846
    B -1.045493 -0.852665 -0.374464
    C -1.419523  1.000000  0.267812
    

**作业5: 熟悉一下fillna的method参数已经他们的用法.**

下面我们来看看怎么用各行的均值来填充缺失值.


```python
print(a.fillna(a.mean()))
```

              a         b         c
    A -1.232508 -1.577756  0.572846
    B -1.045493 -0.852665 -0.374464
    C -1.419523 -1.215210  0.267812
    

注意到, 这个均值是沿着axis=0, 也就是把每行汇总计算得到的.

#### 整数索引

我们这里要谈一个很有意思的问题, 来看看这么个例子:


```python
data = {'a': list(range(1, 4)), 'b': list(range(1, 4)), 'c': list(range(1, 4))}

frame = DataFrame(data, index=['A', 'B', 'C'])

print(frame.loc['A':'B'])
print("--------------------------------------------")
print(frame.iloc[0: 1, 0:1])
print("--------------------------------------------")
print(frame.loc['A':'B', 'a':'b'])
```

       a  b  c
    A  1  1  1
    B  2  2  2
    --------------------------------------------
       a
    A  1
    --------------------------------------------
       a  b
    A  1  1
    B  2  2
    

我们发现, 对于标签, 我们两个端点都是可以取到的, 但是对于数字索引, 是前闭后开.

但是, 如果是对于缩索引, 本身就是整数, 会发生什么呢?


```python
frame = DataFrame(data, index=[1, 2, 3])

print(frame)
```

       a  b  c
    1  1  1  1
    2  2  2  2
    3  3  3  3
    


```python
print(frame.iloc[0: 1])
```

       a  b  c
    1  1  1  1
    


```python
print(frame.loc[1:2])
```

       a  b  c
    1  1  1  1
    2  2  2  2
    

** 我们发现, 这个loc函数, 还是保留了标签, 把这个整数当做标签来处理, 两个端点都是闭的, 而iloc是当做整数索引来处理, 前闭后开.**
