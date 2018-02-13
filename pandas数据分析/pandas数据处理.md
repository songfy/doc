
<h1>目录<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#数据合并" data-toc-modified-id="数据合并-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>数据合并</a></span><ul class="toc-item"><li><span><a href="#实现数据库表join功能" data-toc-modified-id="实现数据库表join功能-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>实现数据库表join功能</a></span></li><li><span><a href="#实现union功能" data-toc-modified-id="实现union功能-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>实现union功能</a></span></li></ul></li><li><span><a href="#数据转换" data-toc-modified-id="数据转换-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>数据转换</a></span><ul class="toc-item"><li><span><a href="#轴旋转" data-toc-modified-id="轴旋转-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>轴旋转</a></span></li><li><span><a href="#数据转换" data-toc-modified-id="数据转换-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>数据转换</a></span><ul class="toc-item"><li><span><a href="#去重" data-toc-modified-id="去重-2.2.1"><span class="toc-item-num">2.2.1&nbsp;&nbsp;</span>去重</a></span></li><li><span><a href="#对某一列运用函数" data-toc-modified-id="对某一列运用函数-2.2.2"><span class="toc-item-num">2.2.2&nbsp;&nbsp;</span>对某一列运用函数</a></span></li><li><span><a href="#重命名行和列名" data-toc-modified-id="重命名行和列名-2.2.3"><span class="toc-item-num">2.2.3&nbsp;&nbsp;</span>重命名行和列名</a></span></li><li><span><a href="#离散化" data-toc-modified-id="离散化-2.2.4"><span class="toc-item-num">2.2.4&nbsp;&nbsp;</span>离散化</a></span></li><li><span><a href="#过滤数据" data-toc-modified-id="过滤数据-2.2.5"><span class="toc-item-num">2.2.5&nbsp;&nbsp;</span>过滤数据</a></span></li><li><span><a href="#转换为onehot表示" data-toc-modified-id="转换为onehot表示-2.2.6"><span class="toc-item-num">2.2.6&nbsp;&nbsp;</span>转换为onehot表示</a></span></li><li><span><a href="#字符串操作" data-toc-modified-id="字符串操作-2.2.7"><span class="toc-item-num">2.2.7&nbsp;&nbsp;</span>字符串操作</a></span></li></ul></li></ul></li></ul></div>

### 数据合并




#### 实现数据库表join功能

当我们有多张表的时候, 经常会遇到的一个问题就是, 如何把这些表关联起来, 我们可以想想我们在数据库的时候,
进场会遇到表连接的问题, 比如join, union等等, 其实这里等同于是在pandas里实现了这些
功能. 首先, 我们来看看这个join在pandas里是怎么实现的.

我们在pandas里主要通过merge来实现数据库的join工作.


```python
from pandas import Series, DataFrame
import pandas as pd
import numpy as np

sep = "---------------------------------------------------------------------"
```


```python
data1 = {"data1": [1, 2, 3, 4, 5], "key":['a', 'b', 'c', 'd', 'e']}
data2 = {"data2": [1, 2, 3,100], "key":['a', 'b', 'c', 'f']}

frame1 = DataFrame(data1)
frame2 = DataFrame(data2)

print(frame1)

print(sep)

print(frame2)

print(sep)

print(pd.merge(frame1, frame2, on="key"))
```

       data1 key
    0      1   a
    1      2   b
    2      3   c
    3      4   d
    4      5   e
    ---------------------------------------------------------------------
       data2 key
    0      1   a
    1      2   b
    2      3   c
    3    100   f
    ---------------------------------------------------------------------
       data1 key  data2
    0      1   a      1
    1      2   b      2
    2      3   c      3
    

注意, 我们默认是inner方式的连接, 对于数据库怎么做连接的, 以及连接的种类, 留作作业.

**作业1: 熟悉数据库连接的方式.**


```python
# 左外连
print(pd.merge(frame1, frame2, on="key", how='left'))

```

       data1 key  data2
    0      1   a    1.0
    1      2   b    2.0
    2      3   c    3.0
    3      4   d    NaN
    4      5   e    NaN
    


```python
# 右外连
print(pd.merge(frame1, frame2, on="key", how='right'))
```

       data1 key  data2
    0    1.0   a      1
    1    2.0   b      2
    2    3.0   c      3
    3    NaN   f    100
    


```python
# 外连接
print(pd.merge(frame1, frame2, on="key", how='outer'))
```

       data1 key  data2
    0    1.0   a    1.0
    1    2.0   b    2.0
    2    3.0   c    3.0
    3    4.0   d    NaN
    4    5.0   e    NaN
    5    NaN   f  100.0
    

我们看到, 这和我们数据库的是一模一样, 我们主要到, on可以指定要关联的列名, 但是我们可能需要关联的列名不同, 这时候我们要分别指定.


```python
data1 = {"data1": [1, 2, 3, 4, 5], "key1":['a', 'b', 'c', 'd', 'e']}
data2 = {"data2": [1, 2, 3,100], "key2":['a', 'b', 'c', 'f']}

frame1 = DataFrame(data1)
frame2 = DataFrame(data2)

print(frame1)

print(sep)

print(frame2)
```

       data1 key1
    0      1    a
    1      2    b
    2      3    c
    3      4    d
    4      5    e
    ---------------------------------------------------------------------
       data2 key2
    0      1    a
    1      2    b
    2      3    c
    3    100    f
    
如果我们要在把key1和key2关联起来, 我们可以怎么做呢? 在sql中, 我们可以用on (key1 = key2), 在pandas中, 我们可以这么做:

```python
print(pd.merge(frame1, frame2, how='inner', left_on='key1', right_on='key2').drop("key1", axis=1))
```

       data1  data2 key2
    0      1      1    a
    1      2      2    b
    2      3      3    c
    

我们发现一个有趣的现象:


```python
print(pd.merge(frame1, frame2, how='outer', left_on='key1', right_on='key2'))
```

       data1 key1  data2 key2
    0    1.0    a    1.0    a
    1    2.0    b    2.0    b
    2    3.0    c    3.0    c
    3    4.0    d    NaN  NaN
    4    5.0    e    NaN  NaN
    5    NaN  NaN  100.0    f
    

我们发现, 这个数据就不一样了, 因为我们是外连接,会保留所有的数据.

多个键做关联也是一样的, 只不过把on改成一个list.

**作业2: 研究多个键关联.**

下面我们来说一个有趣的东西, 我们来看:



```python
data1 = {"data": [1, 2, 3, 4, 5], "key":['a', 'b', 'c', 'd', 'e']}
data2 = {"data": [1, 2, 3,100], "key":['a', 'b', 'c', 'f']}

frame1 = DataFrame(data1)
frame2 = DataFrame(data2)

print(frame1)

print(sep)

print(frame2)

print(sep)

print(pd.merge(frame1, frame2, on='key'))
```

       data key
    0     1   a
    1     2   b
    2     3   c
    3     4   d
    4     5   e
    ---------------------------------------------------------------------
       data key
    0     1   a
    1     2   b
    2     3   c
    3   100   f
    ---------------------------------------------------------------------
       data_x key  data_y
    0       1   a       1
    1       2   b       2
    2       3   c       3
    

我们发现对于列名重复的列, 会自动加上一个后缀, 左边+_x, 右边+_y, 注意这个后缀, 我们是可以自己定义的.


```python
print(pd.merge(frame1, frame2, on='key', suffixes=["-a", "-b"]))
```

       data-a key  data-b
    0       1   a       1
    1       2   b       2
    2       3   c       3
    

然后我们的问题来了, 如果我们要关联的列, 是索引怎么办 , 这个问题有点意思, 但是merge这个函数已经为大家都设计好了, 
我们可以这样搞:


```python
data1 = {"data": [1, 2, 3, 4, 5], "key":['a', 'b', 'c', 'd', 'e']}
data2 = {"data": [1, 2, 3,100], "key":['a', 'b', 'c', 'f']}

frame1 = DataFrame(data1, index=['a', 'b', 'c', 'd', 'e'])
frame2 = DataFrame(data2, index = ['a', 'b', 'c', 'f'])

print(frame1)

print(sep)

print(frame2)

print(sep)

print(pd.merge(frame1, frame2, left_index=True, right_index=True))

print(sep)

frame2 = DataFrame(data2, index = ['1', '2', '3', '4'])

print(frame2)

print(sep)

print(pd.merge(frame1, frame2, left_index=True, right_on='key', how='left'))
```

       data key
    a     1   a
    b     2   b
    c     3   c
    d     4   d
    e     5   e
    ---------------------------------------------------------------------
       data key
    a     1   a
    b     2   b
    c     3   c
    f   100   f
    ---------------------------------------------------------------------
       data_x key_x  data_y key_y
    a       1     a       1     a
    b       2     b       2     b
    c       3     c       3     c
    ---------------------------------------------------------------------
       data key
    1     1   a
    2     2   b
    3     3   c
    4   100   f
    ---------------------------------------------------------------------
      key  data_x key_x  data_y key_y
    1   a       1     a     1.0     a
    2   b       2     b     2.0     b
    3   c       3     c     3.0     c
    4   d       4     d     NaN   NaN
    4   e       5     e     NaN   NaN
    

最后一个例子是把frame1的index和frame2的key连接了起来, 这里我们发现, frame1的索引因为被merge掉了, frame2的索引保留了下来, 同时frame1的key被保留了下来.

我们还有一个函数是join, 他也是实现了按索引关联. 


```python
frame1
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>c</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4</td>
      <td>d</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5</td>
      <td>e</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data</th>
      <th>key</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>c</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>f</td>
    </tr>
  </tbody>
</table>
</div>




```python
frame1.join(frame2, lsuffix="_x", rsuffix="_y")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_x</th>
      <th>key_x</th>
      <th>data_y</th>
      <th>key_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1</td>
      <td>a</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2</td>
      <td>b</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3</td>
      <td>c</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>4</td>
      <td>d</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>5</td>
      <td>e</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



我们发现**调用者的索引被保留了下来**.


```python
# 参数的索引和调用者的列关联在一起
frame2.join(frame1, lsuffix="_x", rsuffix="_y", on="key")
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>data_x</th>
      <th>key_x</th>
      <th>data_y</th>
      <th>key_y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>a</td>
      <td>1.0</td>
      <td>a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>b</td>
      <td>2.0</td>
      <td>b</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>c</td>
      <td>3.0</td>
      <td>c</td>
    </tr>
    <tr>
      <th>4</th>
      <td>100</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



这里设置了on参数, 因此是**调用者的列**和**参数的索引**关联, 最后保留了**调用者的索引**.

#### 实现union功能

上面介绍的都是列关联的,也就是join, 之后我们会看怎么做union. 所谓union就是在纵向上面做连接, 我们可以看到, 这种方式, 可以两张列相同的表拼接起来.


```python
# Series的连接

a = Series([1, 2, 3], index=['a', 'b', 'c'])
b = Series([3, 4], index=['d' , 'e'])
c = Series([6, 7], index=['e', 'f'])

pd.concat([a, b, c])
```




    a    1
    b    2
    c    3
    d    3
    e    4
    e    6
    f    7
    dtype: int64



我们看到, 这样就把这3个Series拼接起来了, 默认是在axis=0上连接的, 但是我们也可以在axis=1上连接, 我们来看看结果怎么样.


```python
pd.concat([a, b, c], axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>f</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



我们看到, 这个相当于是**这3个Series按索引做外连接**. 如果我们要做内连接, 怎么办呢?


```python
pd.concat([b, c], axis=1, join='inner')
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>e</th>
      <td>4</td>
      <td>6</td>
    </tr>
  </tbody>
</table>
</div>



如果要区分从原来哪些地方合并而来的, 我们可以指定keys:


```python
pd.concat([a, b, c], keys=['one', 'two', 'three'])
```




    one     a    1
            b    2
            c    3
    two     d    3
            e    4
    threee  e    6
            f    7
    dtype: int64




```python
pd.concat([a, b, c], axis=1,  keys=['one', 'two', 'three'])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>3.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>e</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>f</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.0</td>
    </tr>
  </tbody>
</table>
</div>



我们发现, 我们的keys在axis=1连接的时候, 变成了列头.

我们下面来看看如果是两个DataFrame, 会怎么样.


```python
f1 = DataFrame(np.arange(6).reshape(3, 2), index=['a', 'b', 'c'], columns=['one', 'two'])

f2 = DataFrame(np.arange(4).reshape(2, 2), index=['c',  'd'], columns=['three', 'four'])

print(f1)

print(sep)

print(f2)
```

       one  two
    a    0    1
    b    2    3
    c    4    5
    ---------------------------------------------------------------------
       three  four
    c      0     1
    d      2     3
    


```python
pd.concat([f1, f2])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>four</th>
      <th>one</th>
      <th>three</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



这就是我们要的union效果, 我们也可以区分出来源


```python
pd.concat([f1, f2], keys=[1, 2])
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>four</th>
      <th>one</th>
      <th>three</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="3" valign="top">1</th>
      <th>a</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>b</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>c</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">2</th>
      <th>c</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>d</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
pd.concat([f1, f2], keys=[1, 2], axis=1)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="2" halign="left">1</th>
      <th colspan="2" halign="left">2</th>
    </tr>
    <tr>
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>three</th>
      <th>four</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.0</td>
      <td>1.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>b</th>
      <td>2.0</td>
      <td>3.0</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>c</th>
      <td>4.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>d</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>3.0</td>
    </tr>
  </tbody>
</table>
</div>



如果我们不想要原来的索引, 而想要重新索引, 我们可以这样来:


```python
pd.concat([f1, f2], ignore_index=True)
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>four</th>
      <th>one</th>
      <th>three</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>4.0</td>
      <td>NaN</td>
      <td>5.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.0</td>
      <td>NaN</td>
      <td>2.0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



总结起来, concat默认就是union的功能, 但是我们可以通过设置axis=1达到按索引关联的功能.

### 数据转换

#### 轴旋转 

这里我们要来聊聊轴旋转的课题, 其中主要用到两个函数:

- **stack** 将列旋转为行
- **unstack** 将行旋转为列


```python
data = DataFrame(np.arange(6).reshape(2, 3), columns=pd.Index(['a', 'b', 'c'], name="column"), index=pd.Index(["one", "two"], name="index"))
print(data)
```

    column  a  b  c
    index          
    one     0  1  2
    two     3  4  5
    

我们来stack一下, 看看会有什么结果:


```python
print(data.stack())

print(sep)

print(data['a']['one'])
print(data.loc['one']['a'])

print(sep)

print(data.stack()['one', 'a'])
```

    index  column
    one    a         0
           b         1
           c         2
    two    a         3
           b         4
           c         5
    dtype: int32
    ---------------------------------------------------------------------
    0
    0
    ---------------------------------------------------------------------
    0
    

我们看到, 我们把每一行都变成了一列, 然后堆了起来, 变成了一个Series.


```python
print(data.stack().unstack())
```

    column  a  b  c
    index          
    one     0  1  2
    two     3  4  5
    

我们来看看, 如果我们把stack后的two, c项给删了, 会怎么样呢?


```python
a = data.stack()
del a['two', 'c']

print(a.unstack())
```

    column    a    b    c
    index                
    one     0.0  1.0  2.0
    two     3.0  4.0  NaN
    

在unstack的时候, 会自动补充NaN值来对齐, 而在stack的时候, 会删除这些NaN值.


```python
a.unstack().stack()
```




    index  column
    one    a         0.0
           b         1.0
           c         2.0
    two    a         3.0
           b         4.0
    dtype: float64



我们发现, 我们在做stack还是unstack的时候, 都是从最内测的轴开始的


```python
b = a.unstack().stack()
print(b)

print(sep)

print(b.unstack())
```

    index  column
    one    a         0.0
           b         1.0
           c         2.0
    two    a         3.0
           b         4.0
    dtype: float64
    ---------------------------------------------------------------------
    column    a    b    c
    index                
    one     0.0  1.0  2.0
    two     3.0  4.0  NaN
    

确实是内侧的column转到了列上面去. 如果我们要转外侧的索引呢, 我们可以指定数字或者列名.


```python
print(b.unstack(0))

print(b.unstack("index"))
```

    index   one  two
    column          
    a       0.0  3.0
    b       1.0  4.0
    c       2.0  NaN
    index   one  two
    column          
    a       0.0  3.0
    b       1.0  4.0
    c       2.0  NaN
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>index</th>
      <th>one</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.0</td>
      <td>2.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.0</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.0</td>
      <td>0.707107</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.0</td>
      <td>3.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.5</td>
      <td>3.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>1.0</td>
      <td>3.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.5</td>
      <td>3.750000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2.0</td>
      <td>4.000000</td>
    </tr>
  </tbody>
</table>
</div>



这样, 行和列就互换了.

强调一下:

**stack: 把行变成列. 我们可以这么理解, 把行堆到了列上.**

**unstack: 把列变成行, 把列反堆到了行上.**

#### 数据转换

##### 去重

去除重复数据, 我们这里主要讲讲怎么能够把重复的数据进行去除


```python
data = DataFrame({'one':[1, 1, 2, 2, 3],'two':[1, 1, 2, 2, 3]})
print(data)
```

       one  two
    0    1    1
    1    1    1
    2    2    2
    3    2    2
    4    3    3
    


```python
print(data.drop_duplicates())
```

       one  two
    0    1    1
    2    2    2
    4    3    3
    


```python
data = DataFrame({'one':[1, 1, 2, 2, 3],'two':[1, 1, 2, 2, 3], 'three':[5, 6, 7, 8, 9]})
print(data)
```

       one  three  two
    0    1      5    1
    1    1      6    1
    2    2      7    2
    3    2      8    2
    4    3      9    3
    




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>three</th>
      <th>two</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>5.00000</td>
      <td>5.000000</td>
      <td>5.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.80000</td>
      <td>7.000000</td>
      <td>1.80000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>0.83666</td>
      <td>1.581139</td>
      <td>0.83666</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00000</td>
      <td>5.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.00000</td>
      <td>6.000000</td>
      <td>1.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.00000</td>
      <td>7.000000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2.00000</td>
      <td>8.000000</td>
      <td>2.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.00000</td>
      <td>9.000000</td>
      <td>3.00000</td>
    </tr>
  </tbody>
</table>
</div>



我们可以按照某一列来进行去重


```python
print(data.drop_duplicates(['one']))
```

       one  three  two
    0    1      5    1
    2    2      7    2
    4    3      9    3
    

去重默认是按照保留最先出现的一个, 我们也可以保留最后出现的一个.

**作业3: 去重, 保留最后出现的一个. **

##### 对某一列运用函数

我们之前提到过, 对于一整列或者一整行, 可以用apply函数, 对于每个元素, 可以用applymap函数, 如果我们要对某一列的

每个元素进行运算, 我们可以用map函数.


```python
data = DataFrame({'one':['a', 'b', 'c'],'two':['e', 'd', 'f']})
print(data)

print(sep)

data['one'] = data['one'].map(str.upper)
print(data)

```

      one two
    0   a   e
    1   b   d
    2   c   f
    ---------------------------------------------------------------------
      one two
    0   A   e
    1   B   d
    2   C   f
    

如果我们只想把a变成大写呢, 我们可以用传入一个map的方法.


```python
data = DataFrame({'one':['a', 'b', 'c'],'two':['e', 'd', 'f']})
print(data)

print(sep)

data['one'] = data['one'].map({'a':'A', 'b':'b', 'c':'c'})
print(data)

```

      one two
    0   a   e
    1   b   d
    2   c   f
    ---------------------------------------------------------------------
      one two
    0   A   e
    1   b   d
    2   c   f
    

在最后一个例子中, 我们发现要提供b和c的值, 太麻烦了, 可以用replace函数:


```python

data['one'] = data['one'].replace('A', 'a')
print(data)
```

      one two
    0   a   e
    1   b   d
    2   c   f
    

##### 重命名行和列名

这里我们来谈谈怎么重命名行或者列的名字. 我们可以用rename函数来完成, 比如我们希望把列名的首字母大写等等, 这个就留作作业.

**作业4: 重命名行名和列名, 把首字母大写.**

##### 离散化

这里会讲一个很有用的技能, 就是离散化, 这个在我们后面处理特征的时候是非常有用的. 离散化主要是用到cut和qcut函数.


```python
a = np.arange(20)
print(a)
```

    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
    


```python
pd.cut(a, 4)  # 这个4将最大值和最小值间分成4等分
```




    [(-0.019, 4.75], (-0.019, 4.75], (-0.019, 4.75], (-0.019, 4.75], (-0.019, 4.75], ..., (14.25, 19.0], (14.25, 19.0], (14.25, 19.0], (14.25, 19.0], (14.25, 19.0]]
    Length: 20
    Categories (4, interval[float64]): [(-0.019, 4.75] < (4.75, 9.5] < (9.5, 14.25] < (14.25, 19.0]]




```python
pd.qcut(a, 4)  # 这个4按照个数分成四等分
```




    [(-0.001, 4.75], (-0.001, 4.75], (-0.001, 4.75], (-0.001, 4.75], (-0.001, 4.75], ..., (14.25, 19.0], (14.25, 19.0], (14.25, 19.0], (14.25, 19.0], (14.25, 19.0]]
    Length: 20
    Categories (4, interval[float64]): [(-0.001, 4.75] < (4.75, 9.5] < (9.5, 14.25] < (14.25, 19.0]]




```python
pd.qcut(a, 4).codes #输出codes
```




    array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3], dtype=int8)



我们也可以按照我们指定的分割点来, 这个留作作业.

**作业5: 按照自己定义的分割点来分割 **


##### 过滤数据

这个小节的功能类似于select中的where语句, 但是要灵活的多, 我们先来看看怎么选出绝对值大于2的行.


```python
data = DataFrame(np.random.randn(10, 10))
print(data)
```

              0         1         2         3         4         5         6  \
    0 -0.064111 -1.237009  0.040219 -0.300265 -0.195558  0.018277 -0.484843   
    1 -0.497673 -0.010135 -1.482219 -0.239210 -0.789893  0.593664  0.345015   
    2 -1.818869  0.613175 -0.165610  0.649670 -1.364698  0.444785 -0.146202   
    3 -0.274151  0.718986  0.321961 -0.416124 -0.275706 -0.738405 -0.260420   
    4  1.980359 -0.429317 -0.964024 -1.474141  0.339342 -0.932012 -0.116387   
    5 -0.518374 -0.224879 -1.517607 -0.079120  0.728408  1.218297  1.191882   
    6 -0.508048  2.010942  1.338983  2.026203 -0.794110 -1.370830  1.364660   
    7  0.855870 -0.804471  0.939610  0.796154  0.467878  0.362091 -1.892815   
    8  1.059561  0.223369  1.098954  1.583732  0.865225 -0.597980 -1.853170   
    9 -0.434388  0.475098 -0.103491 -0.735113  0.823425 -0.905158  0.145539   
    
              7         8         9  
    0 -1.370281  0.112070 -0.387124  
    1  0.049215  0.578946  0.462688  
    2  0.085562  2.906838 -1.059603  
    3 -0.327103  0.504234  0.192760  
    4  0.826963  1.188256  0.590085  
    5 -0.061007  1.955653 -0.984727  
    6 -0.471252  1.067497  0.550022  
    7 -0.503244 -0.288634  1.121110  
    8 -0.650933  0.138730 -0.389139  
    9 -2.078052  0.158038 -0.109184  
    


```python
print(data[np.abs(data) > 2])
```

        0         1   2         3   4   5   6         7         8   9
    0 NaN       NaN NaN       NaN NaN NaN NaN       NaN       NaN NaN
    1 NaN       NaN NaN       NaN NaN NaN NaN       NaN       NaN NaN
    2 NaN       NaN NaN       NaN NaN NaN NaN       NaN  2.906838 NaN
    3 NaN       NaN NaN       NaN NaN NaN NaN       NaN       NaN NaN
    4 NaN       NaN NaN       NaN NaN NaN NaN       NaN       NaN NaN
    5 NaN       NaN NaN       NaN NaN NaN NaN       NaN       NaN NaN
    6 NaN  2.010942 NaN  2.026203 NaN NaN NaN       NaN       NaN NaN
    7 NaN       NaN NaN       NaN NaN NaN NaN       NaN       NaN NaN
    8 NaN       NaN NaN       NaN NaN NaN NaN       NaN       NaN NaN
    9 NaN       NaN NaN       NaN NaN NaN NaN -2.078052       NaN NaN
    

完了, 居然是这幅德行


```python
np.abs(data) > 2
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>5</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>6</th>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>8</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
    </tr>
    <tr>
      <th>9</th>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>False</td>
      <td>True</td>
      <td>False</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
</div>



我们发现, false的这些地方, 都被设为NaN, 我们不想要这些false的数据, 我们只需要存在一个大于2的行, 我们看看apply函数行不行呢?


```python
data[(np.abs(data) > 2).apply(lambda x: x.name if x.sum() > 0 else None, axis=1).notnull()]
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-1.818869</td>
      <td>0.613175</td>
      <td>-0.165610</td>
      <td>0.649670</td>
      <td>-1.364698</td>
      <td>0.444785</td>
      <td>-0.146202</td>
      <td>0.085562</td>
      <td>2.906838</td>
      <td>-1.059603</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.508048</td>
      <td>2.010942</td>
      <td>1.338983</td>
      <td>2.026203</td>
      <td>-0.794110</td>
      <td>-1.370830</td>
      <td>1.364660</td>
      <td>-0.471252</td>
      <td>1.067497</td>
      <td>0.550022</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.434388</td>
      <td>0.475098</td>
      <td>-0.103491</td>
      <td>-0.735113</td>
      <td>0.823425</td>
      <td>-0.905158</td>
      <td>0.145539</td>
      <td>-2.078052</td>
      <td>0.158038</td>
      <td>-0.109184</td>
    </tr>
  </tbody>
</table>
</div>



这样看上去实在是太复杂了, 其实可以简化


```python
(np.abs(data) > 2).any(1)
```




    0    False
    1    False
    2     True
    3    False
    4    False
    5    False
    6     True
    7    False
    8    False
    9     True
    dtype: bool




```python
data[(np.abs(data) > 2).any(1)] # 有一个真就是真
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>-1.818869</td>
      <td>0.613175</td>
      <td>-0.165610</td>
      <td>0.649670</td>
      <td>-1.364698</td>
      <td>0.444785</td>
      <td>-0.146202</td>
      <td>0.085562</td>
      <td>2.906838</td>
      <td>-1.059603</td>
    </tr>
    <tr>
      <th>6</th>
      <td>-0.508048</td>
      <td>2.010942</td>
      <td>1.338983</td>
      <td>2.026203</td>
      <td>-0.794110</td>
      <td>-1.370830</td>
      <td>1.364660</td>
      <td>-0.471252</td>
      <td>1.067497</td>
      <td>0.550022</td>
    </tr>
    <tr>
      <th>9</th>
      <td>-0.434388</td>
      <td>0.475098</td>
      <td>-0.103491</td>
      <td>-0.735113</td>
      <td>0.823425</td>
      <td>-0.905158</td>
      <td>0.145539</td>
      <td>-2.078052</td>
      <td>0.158038</td>
      <td>-0.109184</td>
    </tr>
  </tbody>
</table>
</div>



这里这个any(1)相当于是apply(lambda x: x.name if x.sum() > 0 else None, axis=1).notnull()

##### 转换为onehot表示

下面我们来提一个东西, 就是怎么将数据转换为onehot的表示.


```python
data = DataFrame({'one': np.arange(20), 'two': np.arange(20)})
data.join(pd.get_dummies(pd.cut(data['one'],  4).values.codes, prefix="one_"))

```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>one</th>
      <th>two</th>
      <th>one__0</th>
      <th>one__1</th>
      <th>one__2</th>
      <th>one__3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5</td>
      <td>5</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6</td>
      <td>6</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>10</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>13</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14</td>
      <td>14</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15</td>
      <td>15</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16</td>
      <td>16</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17</td>
      <td>17</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18</td>
      <td>18</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



是不是超级简单.

##### 字符串操作

我们来看看字符串的操作, 其实主要还是正则表达式, 我们来看一个例子:



```python
data = DataFrame({'a':["xiaoming@sina.com", "xiaozhang@gmail.com", "xiaohong@qq.com"], 'b':[1, 2, 3]})
print(data)
```

                         a  b
    0    xiaoming@sina.com  1
    1  xiaozhang@gmail.com  2
    2      xiaohong@qq.com  3
    


```python
import re
pattern = r'([A-Z]+)@([A-Z]+)\.([A-Z]{2,4})'

data['a'].str.findall(pattern, flags=re.IGNORECASE).str[0].str[1]


```




    0     sina
    1    gmail
    2       qq
    Name: a, dtype: object



我们看到, 我们可以用python的正则表达式来处理字符串问题.

**作业6: 熟悉python正则表达式. http://www.runoob.com/python3/python3-reg-expressions.html**
