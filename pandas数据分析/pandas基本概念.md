
<h1>目录<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#pandas基本概念" data-toc-modified-id="pandas基本概念-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>pandas基本概念</a></span><ul class="toc-item"><li><span><a href="#pandas数据结构剖析" data-toc-modified-id="pandas数据结构剖析-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>pandas数据结构剖析</a></span><ul class="toc-item"><li><span><a href="#Series" data-toc-modified-id="Series-1.1.1"><span class="toc-item-num">1.1.1&nbsp;&nbsp;</span>Series</a></span></li><li><span><a href="#DataFrame" data-toc-modified-id="DataFrame-1.1.2"><span class="toc-item-num">1.1.2&nbsp;&nbsp;</span>DataFrame</a></span></li><li><span><a href="#索引" data-toc-modified-id="索引-1.1.3"><span class="toc-item-num">1.1.3&nbsp;&nbsp;</span>索引</a></span></li></ul></li></ul></li></ul></div>

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
    0  0.285786  2.071394  1.742614
    1 -1.337236  0.310217  0.512296
    2  0.382217 -0.555744  0.601913
    -----------------------------------------------------------------
              a         b         c
    A  0.285786  2.071394  1.742614
    B -1.337236  0.310217  0.512296
    C  0.382217 -0.555744  0.601913
    

我们用索引的方式, 可以获取到某一列, 作为Series.


```python
print(frame['a'])
```

    A    0.285786
    B   -1.337236
    C    0.382217
    Name: a, dtype: float64
    

我们发现, 这个Series的名字, 就是列名.

如果我们要修改DataFrame的某个值, 怎么改呢? 我们先来看看怎么修改某一列的值.


```python
# 修改某一个列的值
frame['a'] = [1,1,1]
print(frame)
```

       a         b         c
    A  1  2.071394  1.742614
    B  1  0.310217  0.512296
    C  1 -0.555744  0.601913
    


```python
# 如果我们list的长度不符合, 就会报错
frame['a'] = [1,1]
```


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    <ipython-input-13-54000a6faf10> in <module>()
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

如果我们要新加一列, 可以这样写:


```python
frame['d'] = s
print(frame)
```

如果我们要删除一列, 可以用del


```python
print(frame.columns)
del frame['d']
print(frame.columns)
```

我们还可以用字典来初始化我们的DataFrame:


```python
data = {'a':{'A':1,'B':2}, 'b':{'A':3,'B':4}}
frame = DataFrame(data)
print(frame)
print(frame.columns)
print(frame.index)
print(frame.values)
```

有Series的字典也可以初始化DataFrame, 我们这里不演示了, 留作作业.

**作业1: 用Series的字典初始化DataFrame.**

#### 索引

下面我们来里聊聊索引, 这个也是pandas中比较基本的一个概念.


```python
index = frame.index
colunms = frame.columns
print(type(colunms))
print(type(index))
```

    <class 'pandas.core.indexes.base.Index'>
    <class 'pandas.core.indexes.base.Index'>
    

我们可以把索引理解为 ** 不可变的hashset **


```python
print(index)
```

    Index(['A', 'B', 'C'], dtype='object')
    


```python
index[1]='D'
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-16-e57751cb899e> in <module>()
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
