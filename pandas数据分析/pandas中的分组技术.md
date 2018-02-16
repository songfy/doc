
<h1>目录<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#分组操作" data-toc-modified-id="分组操作-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>分组操作</a></span><ul class="toc-item"><li><span><a href="#按照列进行分组" data-toc-modified-id="按照列进行分组-1.1"><span class="toc-item-num">1.1&nbsp;&nbsp;</span>按照列进行分组</a></span></li><li><span><a href="#按照字典进行分组" data-toc-modified-id="按照字典进行分组-1.2"><span class="toc-item-num">1.2&nbsp;&nbsp;</span>按照字典进行分组</a></span></li><li><span><a href="#根据函数进行分组" data-toc-modified-id="根据函数进行分组-1.3"><span class="toc-item-num">1.3&nbsp;&nbsp;</span>根据函数进行分组</a></span></li><li><span><a href="#按照list组合" data-toc-modified-id="按照list组合-1.4"><span class="toc-item-num">1.4&nbsp;&nbsp;</span>按照list组合</a></span></li><li><span><a href="#按照索引级别进行分组" data-toc-modified-id="按照索引级别进行分组-1.5"><span class="toc-item-num">1.5&nbsp;&nbsp;</span>按照索引级别进行分组</a></span></li></ul></li><li><span><a href="#分组运算" data-toc-modified-id="分组运算-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>分组运算</a></span><ul class="toc-item"><li><span><a href="#agg" data-toc-modified-id="agg-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>agg</a></span></li><li><span><a href="#transform" data-toc-modified-id="transform-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>transform</a></span></li><li><span><a href="#apply" data-toc-modified-id="apply-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>apply</a></span></li></ul></li><li><span><a href="#利用groupby技术多进程处理DataFrame" data-toc-modified-id="利用groupby技术多进程处理DataFrame-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>利用groupby技术多进程处理DataFrame</a></span></li></ul></div>

我们在这里要讲一个很常用的技术, 就是所谓的分组技术, 这个在数据库中是非常常用的, 要去求某些分组的统计量, 那么我们需要知道在pandas里面, 这些分组技术是怎么实现的.

### 分组操作

我们这里要来聊聊在pandas中实现分组运算, 大致上可以按照列, 字典或者Series, 函数, 索引级别进行分组, 我们会逐渐来介绍.



#### 按照列进行分组


```python
import pandas as pd
from pandas import DataFrame, Series
import numpy as np

sep = "---------------------------------------------------------------------------"
```


```python
data = DataFrame({"key1": ['a', 'a', 'b', 'b', 'a'], "key2": ['one', 'two', 'one', 'two', 'one'], 'data1': np.random.randn(5), 'data2': np.random.randn(5)})
print(data)
```

          data1     data2 key1 key2
    0  0.733951  0.000379    a  one
    1  1.039029  0.852930    a  two
    2  0.921413 -1.644942    b  one
    3  0.294560  0.521525    b  two
    4  0.286072 -0.074574    a  one
    

data1按照key1分组为:


```python
groups = data['data1'].groupby(data['key1'])
```

我们发现得到了一个SeriesGroupBy 对象, 现在我们对这个对象进行**迭代**:


```python
for name, group in groups:
    print(name)
    print(sep)
    print(group)
    print(sep)
```

    a
    ---------------------------------------------------------------------------
    0    0.733951
    1    1.039029
    4    0.286072
    Name: data1, dtype: float64
    ---------------------------------------------------------------------------
    b
    ---------------------------------------------------------------------------
    2    0.921413
    3    0.294560
    Name: data1, dtype: float64
    ---------------------------------------------------------------------------
    

我们发现, **groups有(key, Series)对组成, key根据什么来分组的元素, Series(DataFrame)是分组的元素, Series(DataFrame)的name还是原来的列名**.

对你分组进行迭代, 用:

**for name, group in groups**


```python
groups = data.groupby(data['key1'])
for name, group in groups:
    print(name)
    print(sep)
    print(group)
    print(sep)
```

    a
    ---------------------------------------------------------------------------
          data1     data2 key1 key2
    0  0.733951  0.000379    a  one
    1  1.039029  0.852930    a  two
    4  0.286072 -0.074574    a  one
    ---------------------------------------------------------------------------
    b
    ---------------------------------------------------------------------------
          data1     data2 key1 key2
    2  0.921413 -1.644942    b  one
    3  0.294560  0.521525    b  two
    ---------------------------------------------------------------------------
    

**groupby就是按照某个值来分组, 无论是对series还是dataframe, 都成立.**

我们可以在分好组的对象上调用统计函数.


```python
data.groupby(data['key1']).mean()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key1</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>a</th>
      <td>0.686351</td>
      <td>0.259578</td>
    </tr>
    <tr>
      <th>b</th>
      <td>0.607986</td>
      <td>-0.561709</td>
    </tr>
  </tbody>
</table>
</div>



在每个分组上分别对每个每一列求均值, 如果是非数字列, 或默认剔除.

**作业1:在每个分组上分别对每个每一行求均值.**

提示: data.groupby(data['key1']).mean(axis=1)是行不通的.

**对于多个列进行分组, 分组的key是对应分组元素的元组.**

**作业2:对DataFrame用多个列进行分组.**

下面其我们来看一个语法糖:


```python
data.groupby([data['key1'], data['key2']])
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x000001D080230278>



它等价于:


```python
data.groupby(['key1', 'key2'])
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x000001D080230630>



我们来验证一下:


```python
groups =data.groupby([data['key1'], data['key2']])
for name, group in groups:
    print(name)
    print(sep)
    print(group)
    print(sep)
```

    ('a', 'one')
    ---------------------------------------------------------------------------
          data1     data2 key1 key2
    0  0.733951  0.000379    a  one
    4  0.286072 -0.074574    a  one
    ---------------------------------------------------------------------------
    ('a', 'two')
    ---------------------------------------------------------------------------
          data1    data2 key1 key2
    1  1.039029  0.85293    a  two
    ---------------------------------------------------------------------------
    ('b', 'one')
    ---------------------------------------------------------------------------
          data1     data2 key1 key2
    2  0.921413 -1.644942    b  one
    ---------------------------------------------------------------------------
    ('b', 'two')
    ---------------------------------------------------------------------------
         data1     data2 key1 key2
    3  0.29456  0.521525    b  two
    ---------------------------------------------------------------------------
    


```python
groups = data.groupby(['key1', 'key2'])
for name, group in groups:
    print(name)
    print(sep)
    print(group)
    print(sep)
```

    ('a', 'one')
    ---------------------------------------------------------------------------
          data1     data2 key1 key2
    0  0.733951  0.000379    a  one
    4  0.286072 -0.074574    a  one
    ---------------------------------------------------------------------------
    ('a', 'two')
    ---------------------------------------------------------------------------
          data1    data2 key1 key2
    1  1.039029  0.85293    a  two
    ---------------------------------------------------------------------------
    ('b', 'one')
    ---------------------------------------------------------------------------
          data1     data2 key1 key2
    2  0.921413 -1.644942    b  one
    ---------------------------------------------------------------------------
    ('b', 'two')
    ---------------------------------------------------------------------------
         data1     data2 key1 key2
    3  0.29456  0.521525    b  two
    ---------------------------------------------------------------------------
    

我们发现输出结果是一模一样, 总结一下:

**data.groupby([data['key1'], data['key2']])等价于data.groupby(['key1', 'key2'])**

进一步:

**data['data1'].groupby([data['key1'], data['key2']])等价于data.groupby(['key1', 'key2'])['data1']**

**作业3: 验证data['data1'].groupby([data['key1'], data['key2']])等价于data.groupby(['key1', 'key2'])['data1'].**


```python
data.groupby(['key1', 'key2'])['data1']
```




    <pandas.core.groupby.SeriesGroupBy object at 0x000001D0FCD95D68>




```python
data.groupby(['key1', 'key2'])[['data1']]
```




    <pandas.core.groupby.DataFrameGroupBy object at 0x000001D080232898>



我不知道大家发现没有, 这两个返回的数据类型是有区别的, 我们仔细来看看:


```python
data[['data1']] # 这是一个DataFrame
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
      <th>data1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.733951</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.039029</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.921413</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.294560</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.286072</td>
    </tr>
  </tbody>
</table>
</div>




```python
data['data1'] # 这是一个Series
```




    0    0.733951
    1    1.039029
    2    0.921413
    3    0.294560
    4    0.286072
    Name: data1, dtype: float64



那么这里的区别就不言而喻了吧


```python
groups = data.groupby(['key1', 'key2'])[['data1']]

for name, group in groups:
    print(name)
    print(sep)
    print(group)
    print(sep)
```

    ('a', 'one')
    ---------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    ---------------------------------------------------------------------------
    ('a', 'two')
    ---------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    ---------------------------------------------------------------------------
    ('b', 'one')
    ---------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    ---------------------------------------------------------------------------
    ('b', 'two')
    ---------------------------------------------------------------------------
    <class 'pandas.core.frame.DataFrame'>
    ---------------------------------------------------------------------------
    

结果是一样的.


```python
data.groupby(['key1', 'key2'])[['data1']].mean()
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
      <th>data1</th>
    </tr>
    <tr>
      <th>key1</th>
      <th>key2</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">a</th>
      <th>one</th>
      <td>0.510012</td>
    </tr>
    <tr>
      <th>two</th>
      <td>1.039029</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">b</th>
      <th>one</th>
      <td>0.921413</td>
    </tr>
    <tr>
      <th>two</th>
      <td>0.294560</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.groupby(['key1', 'key2'])['data1'].mean()
```




    key1  key2
    a     one     0.510012
          two     1.039029
    b     one     0.921413
          two     0.294560
    Name: data1, dtype: float64



在做数据聚合的时候就发现了不同,

**[['data1']]得到的是一个DataFrame, 而['data1']得到的是Series.**

#### 按照字典进行分组

我们来看一个按照字典进行分组的例子:


```python
data = DataFrame(np.random.randn(5, 5), columns=['a', 'b', 'c', 'd', 'e'], index=['joe', 'steve', 'wes', 'jim', 'Travis'])
```


```python
data
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>joe</th>
      <td>-0.089597</td>
      <td>1.239307</td>
      <td>2.173063</td>
      <td>-0.519295</td>
      <td>-1.783812</td>
    </tr>
    <tr>
      <th>steve</th>
      <td>0.539109</td>
      <td>0.724553</td>
      <td>-0.041899</td>
      <td>0.787494</td>
      <td>0.394633</td>
    </tr>
    <tr>
      <th>wes</th>
      <td>-0.055417</td>
      <td>0.384068</td>
      <td>-0.594006</td>
      <td>-0.451587</td>
      <td>0.722761</td>
    </tr>
    <tr>
      <th>jim</th>
      <td>-0.056767</td>
      <td>0.398863</td>
      <td>2.140669</td>
      <td>-1.060791</td>
      <td>-0.953756</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>0.245142</td>
      <td>-0.468819</td>
      <td>-0.863372</td>
      <td>-0.151966</td>
      <td>1.185567</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 定义一个分组的字典, a, b, c --> red, d, e --> blue
mapping = {'a':'red', 'b':'red', 'c': 'red', 'd':'blue', 'e': 'blue'}
data.groupby(mapping, axis=1).mean()   # 对每一个分组求平均
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
      <th>blue</th>
      <th>red</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>joe</th>
      <td>-1.151554</td>
      <td>1.107591</td>
    </tr>
    <tr>
      <th>steve</th>
      <td>0.591063</td>
      <td>0.407255</td>
    </tr>
    <tr>
      <th>wes</th>
      <td>0.135587</td>
      <td>-0.088452</td>
    </tr>
    <tr>
      <th>jim</th>
      <td>-1.007273</td>
      <td>0.827589</td>
    </tr>
    <tr>
      <th>Travis</th>
      <td>0.516800</td>
      <td>-0.362350</td>
    </tr>
  </tbody>
</table>
</div>



**作业4:自己设计一个index的mapping, 按axis=0进行分组.**

####  根据函数进行分组

话不多说, 直接来看例子:


```python
data.groupby(len).mean()
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>3</th>
      <td>-0.067260</td>
      <td>0.674079</td>
      <td>1.239909</td>
      <td>-0.677224</td>
      <td>-0.671602</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.539109</td>
      <td>0.724553</td>
      <td>-0.041899</td>
      <td>0.787494</td>
      <td>0.394633</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.245142</td>
      <td>-0.468819</td>
      <td>-0.863372</td>
      <td>-0.151966</td>
      <td>1.185567</td>
    </tr>
  </tbody>
</table>
</div>



我们发现, **字典和函数都是作用到索引上的.**

#### 按照list组合

这个例子非常简单:


```python
data.groupby(['1', '1', '1', '2', '2']).mean()
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
      <th>a</th>
      <th>b</th>
      <th>c</th>
      <th>d</th>
      <th>e</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.131365</td>
      <td>0.782643</td>
      <td>0.512386</td>
      <td>-0.061130</td>
      <td>-0.222139</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.094188</td>
      <td>-0.034978</td>
      <td>0.638649</td>
      <td>-0.606378</td>
      <td>0.115905</td>
    </tr>
  </tbody>
</table>
</div>



他会自动判断是按照列还是list.

#### 按照索引级别进行分组

**作业5: 自己学习按索引级别进行分组.**

### 分组运算

分组运算主要设计到3个函数, agg, transform和apply. 


我们一个一个来看.

#### agg


```python
data = DataFrame({"key1": ['a', 'a', 'b', 'b', 'a'], "key2": ['one', 'two', 'one', 'two', 'one'], 'data1': np.random.randn(5), 'data2': np.random.randn(5)})
```


```python
data
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
      <th>data1</th>
      <th>data2</th>
      <th>key1</th>
      <th>key2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.441278</td>
      <td>-0.848457</td>
      <td>a</td>
      <td>one</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.843375</td>
      <td>-0.522482</td>
      <td>a</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.435176</td>
      <td>-0.191682</td>
      <td>b</td>
      <td>one</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.700772</td>
      <td>-0.832993</td>
      <td>b</td>
      <td>two</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.430386</td>
      <td>-1.910834</td>
      <td>a</td>
      <td>one</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.groupby("key2").agg(np.mean)
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>-0.808095</td>
      <td>-0.983658</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.428699</td>
      <td>-0.677738</td>
    </tr>
  </tbody>
</table>
</div>



当然, 这个等价于:


```python
data.groupby("key2").mean()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>-0.808095</td>
      <td>-0.983658</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.428699</td>
      <td>-0.677738</td>
    </tr>
  </tbody>
</table>
</div>



**原理: 聚合函数会在group后的每个切片上(group后的每一行或每一列)计算出值.**

我们也可以自定义函数:


```python
data.groupby("key2").agg(lambda x: x.max() - x.min())
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
      <th>data1</th>
      <th>data2</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>1.876454</td>
      <td>1.719153</td>
    </tr>
    <tr>
      <th>two</th>
      <td>4.544147</td>
      <td>0.310511</td>
    </tr>
  </tbody>
</table>
</div>



他会在每个分组的每个列上调用这个函数.


```python
data.groupby("key2").agg([np.mean, np.max,np.min])
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
      <th colspan="3" halign="left">data1</th>
      <th colspan="3" halign="left">data2</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>amax</th>
      <th>amin</th>
      <th>mean</th>
      <th>amax</th>
      <th>amin</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>one</th>
      <td>-0.808095</td>
      <td>0.441278</td>
      <td>-1.435176</td>
      <td>-0.983658</td>
      <td>-0.191682</td>
      <td>-1.910834</td>
    </tr>
    <tr>
      <th>two</th>
      <td>-0.428699</td>
      <td>1.843375</td>
      <td>-2.700772</td>
      <td>-0.677738</td>
      <td>-0.522482</td>
      <td>-0.832993</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.groupby("key2").agg([("平均值:", np.mean), ("最大值",np.max), ("最小值",np.min)]).rename({"one": "第一组", "two":"第二组"})
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
      <th colspan="3" halign="left">data1</th>
      <th colspan="3" halign="left">data2</th>
    </tr>
    <tr>
      <th></th>
      <th>平均值:</th>
      <th>最大值</th>
      <th>最小值</th>
      <th>平均值:</th>
      <th>最大值</th>
      <th>最小值</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>第一组</th>
      <td>-0.808095</td>
      <td>0.441278</td>
      <td>-1.435176</td>
      <td>-0.983658</td>
      <td>-0.191682</td>
      <td>-1.910834</td>
    </tr>
    <tr>
      <th>第二组</th>
      <td>-0.428699</td>
      <td>1.843375</td>
      <td>-2.700772</td>
      <td>-0.677738</td>
      <td>-0.522482</td>
      <td>-0.832993</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 对不同列用不同的分组函数 
data.groupby("key2").agg({"data1":[("平均值:", np.mean), ("最大值",np.max)], "data2":[("最小值",np.min)]}).rename({"one": "第一组", "two":"第二组"})
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
      <th>data2</th>
      <th colspan="2" halign="left">data1</th>
    </tr>
    <tr>
      <th></th>
      <th>最小值</th>
      <th>平均值:</th>
      <th>最大值</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>第一组</th>
      <td>-1.910834</td>
      <td>-0.808095</td>
      <td>0.441278</td>
    </tr>
    <tr>
      <th>第二组</th>
      <td>-0.832993</td>
      <td>-0.428699</td>
      <td>1.843375</td>
    </tr>
  </tbody>
</table>
</div>



#### transform

transform是一个矢量化的函数, 如果最后我们得到的值和分组切片不一致, 会进行广播:



```python
data
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
      <th>data1</th>
      <th>data2</th>
      <th>key1</th>
      <th>key2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.441278</td>
      <td>-0.848457</td>
      <td>a</td>
      <td>one</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.843375</td>
      <td>-0.522482</td>
      <td>a</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-1.435176</td>
      <td>-0.191682</td>
      <td>b</td>
      <td>one</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.700772</td>
      <td>-0.832993</td>
      <td>b</td>
      <td>two</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.430386</td>
      <td>-1.910834</td>
      <td>a</td>
      <td>one</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.groupby("key1").transform(np.mean)
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.284756</td>
      <td>-1.093924</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.284756</td>
      <td>-1.093924</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-2.067974</td>
      <td>-0.512338</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-2.067974</td>
      <td>-0.512338</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.284756</td>
      <td>-1.093924</td>
    </tr>
  </tbody>
</table>
</div>



仔细看, 0,1, 4一组, 2,3一组, 发生了广播.

现在有个需求,按分组减去均值.


```python
data.groupby("key1").transform(lambda x: x - x.mean())
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.156523</td>
      <td>0.245468</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.558619</td>
      <td>0.571442</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.632798</td>
      <td>0.320656</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.632798</td>
      <td>-0.320656</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-1.715142</td>
      <td>-0.816910</td>
    </tr>
  </tbody>
</table>
</div>



a, b分组的各列都减去了他们的均值, 不信, 来看:


```python
data.groupby("key1").transform(lambda x: x - x.mean()).groupby([1, 1, 0,0, 1]).mean()
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
      <th>data1</th>
      <th>data2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.110223e-16</td>
      <td>-5.551115e-17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.401487e-17</td>
      <td>-1.110223e-16</td>
    </tr>
  </tbody>
</table>
</div>



#### apply

这个函数是transform的加强版, transform只能返回和原来切片大小一样大的, 但apply是可以任意的. 其实我们之前就用过apply函数, 我们知道, apply是作用在列(行)上的, applymap是作用在函数上的.


```python
data = DataFrame({"key1": ['a', 'a', 'b', 'b', 'a'], "key2": ['one', 'two', 'one', 'two', 'one'], 'data1': np.random.randn(5), 'data2': np.random.randn(5)})
```


```python
data
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
      <th>data1</th>
      <th>data2</th>
      <th>key1</th>
      <th>key2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.312694</td>
      <td>0.073574</td>
      <td>a</td>
      <td>one</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.902065</td>
      <td>-0.854249</td>
      <td>a</td>
      <td>two</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.440915</td>
      <td>0.228551</td>
      <td>b</td>
      <td>one</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.406243</td>
      <td>-0.878505</td>
      <td>b</td>
      <td>two</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.812926</td>
      <td>-0.114598</td>
      <td>a</td>
      <td>one</td>
    </tr>
  </tbody>
</table>
</div>



如果我们要找出one, 和two分组中选出data2最大的前两个呢?


```python
data.groupby('key2').apply(lambda x: x.sort_values(by='data2')[-2:])
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
      <th>data1</th>
      <th>data2</th>
      <th>key1</th>
      <th>key2</th>
    </tr>
    <tr>
      <th>key2</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="2" valign="top">one</th>
      <th>0</th>
      <td>-0.312694</td>
      <td>0.073574</td>
      <td>a</td>
      <td>one</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.440915</td>
      <td>0.228551</td>
      <td>b</td>
      <td>one</td>
    </tr>
    <tr>
      <th rowspan="2" valign="top">two</th>
      <th>3</th>
      <td>-0.406243</td>
      <td>-0.878505</td>
      <td>b</td>
      <td>two</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.902065</td>
      <td>-0.854249</td>
      <td>a</td>
      <td>two</td>
    </tr>
  </tbody>
</table>
</div>



去掉group层次索引:


```python
data.groupby('key2', group_keys=False).apply(lambda x: x.sort_values(by='data2')[-2:])
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
      <th>data1</th>
      <th>data2</th>
      <th>key1</th>
      <th>key2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.312694</td>
      <td>0.073574</td>
      <td>a</td>
      <td>one</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.440915</td>
      <td>0.228551</td>
      <td>b</td>
      <td>one</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.406243</td>
      <td>-0.878505</td>
      <td>b</td>
      <td>two</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.902065</td>
      <td>-0.854249</td>
      <td>a</td>
      <td>two</td>
    </tr>
  </tbody>
</table>
</div>



**总结一下: apply就是把分完组的切片挨个(按行, 按列, 或者整体)调用我们的函数, 最后再把结果合并起来.**

### 利用groupby技术多进程处理DataFrame

我们这里要教大家用一种groupby技术, 来实现对DataFrame并行处理.

pip install joblib

因为我们windows系统的限制, 我们的代码是在linux上运行的:


```python

import math
from joblib import Parallel, delayed
from pandas import DataFrame
import pandas as pd
import numpy as np
import time

begin = time.time()
test = DataFrame(np.random.randn(10000000, 10))
test_other = test.copy()
groups = test.groupby(lambda x: x % 8)

def func(x):
    return x.applymap(lambda y: math.pow(y, 4))

pd.concat(Parallel(n_jobs=8)(delayed(func)(group) for name, group in groups))
print(time.time() - begin)


begin = time.time()
test_other.applymap(lambda x: math.pow(x, 4))
print(time.time() - begin)

```

运算结果为:
23.35878014564514
62.76386260986328


速度大概提升了2.5倍, 还是很不错的.
