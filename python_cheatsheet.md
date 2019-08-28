## Python Notes for future reference

#### Versions
Python 3.0 introduced many backwards-incompatible changes to the language, so code written for 2.7 may not work under 3.5 and vice versa.

#### Types
1. In python, you don't have x++, x-- operators.
2. Python also has built-in types for complex numbers.
3. and, or instead of "&&" and "||".
4. String objects have a bunch of useful methods.

#### Containers
Build-in container types: lists, dictionaries, sets and tuples
(Container type are all python objects that contain other object like list or dict).

##### Lists
A list is the Python equivalent of an array, but is resizable and can contain elements of different types. <br/>
Ex: <br/>
xs = [3, 1, 2] <br/>
xs[-1] # Negative element count from the end of the list <br/>
xs.append("bar") <br/>
xs.pop() # Remove and return the last element of the list

##### Slicing
In addition to accessing list elements one at a time, Python provides slicing. <br/>
Ex: <br/>
nums[2:4] # only second and third elements will be shown

##### Built-in enumerate
animals = ['cat', 'dog', 'monkey'] <br/>
for idx, animal in enumerate(animals):

##### List comprehensions:
Ex: <br/>
nums = [0, 1, 2, 3, 4] <br/>
squares = [x ** 2 for x in nums] <br/>
print(squares)

##### Dictionaries
A dictionary stores (key, value) pairs, similar to a Map in Java. <br/>
Ex: <br/>
d = {'person': 2, 'cat': 4, 'spider': 8} <br/>
<pre>
for animal, legs in d.iems(): <br/>
	print('A %s has %d legs' % (animal, legs))
</pre>
##### Dictionary comprehensions

##### Sets
A set is an unordered collection of distinct elements. <br/>
Ex: <br/>
animals = {'cat', 'dog'} <br/>
print('cat' in animals) # Check if an element is in the set <br/>
animals.add('fish') <br/>
animals.remove('cat')

##### Tuple
A tuple is an (immutable) ordered list of values. A tuple is in many
ways similar to a list; Tuples can be used as keys in dictionaries and as elements of sets, while lists cannot.

##### Functions
We will often define functions to take optional keyword arguments.

##### Classes
ex: <br/>
> class Greeter(object):
	# Constructor
    def _init_(self,name):
    	self.name = name # Create an instance variable
>
    # Instance method
    def greet(self, loud = False):
    	if loud:
        	print('HELLO, %s!' % self.name.upper())
		else:
        	print('Hello, %s' % self.name)

g = Greeter('Fred') <br/>
g.greet() <br/>
g.greet(loud=True)

##### Numpy
Numpy is the core library for scientific computing in Python.
It provides a high-performance multidimensional array object, and tools for working with these arrays.

###### Arrays
1. A numpy array is a grid of values, all of the same type and is indexed by a tuple of non-negative integers.
2. The number of dimensions is the rank of the array.
3. The shape of an array is a tuple of integers giving the  size of the array along each dimension.

We can initialize numpy arrays from nested Python lists and access elements using Square brackets:
> import numpy as np <br/>

> a = np.array([1, 2, 3]) # Create a rank 1 array <br/>
> print(type(a)) <br/>
> print(a.shape)

> a = np.zeros((2,2)) # Create an array of all zeros <br/>
> b = np.ones((1,2)) # Create an array of all ones <br/>
> c = np.full((2,2), 7) # create an constant array all 7s <br/>
> d = np.eye(2) # Create a 2*2 identity matrix <br/>
> e = np.random.random((2,2)) # Create an array filled with random values

###### Array indexing
Numpy offers several ways to index into arrays. <br/>
Slicing: numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array.

> a = np.array([1,2,3,4],[5,6,7,8],[9,10,11,12]) <br/>
> b = a[:2, 1:3] # [[2 3] [6 7]] <br/>

Mixing integer indexing with slices yields an array lower rank, while using only slices yields an array of the same rank as the original rank
> row_r1 = a[1, :] # Rank 1  when view of the second row of a <br/>
> row_r2 = a[1:2, :] # Rank 2 view of the second row of a <br/>
> (4,) and (1,4)

Same distinction when accessing columns of an array
> col_r1 = a[:,1]
> col_r2 = a[:,1:2]
> (3,) and (3,1)

Integer array indexing: When you index into numpy arrays using slicing, the resulting array will always be a subarray of the original array. In contrast, integer array indexing allows you to construct arbitrary arrays using the data from another array.

One useful trick is selecting or mutating one element from each row of a matrix:
> a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]) <br/>
> b = np.array([0, 2, 0, 1]) <br/>
> print(a[np.arange(4), b])

boolean array indexing: lets you pick out arbitrary elements of an array. Frequently this type of indexing is used to select the elements of an array that satisfy some condition. Ex:

> a = np.array([1,2], [3,4], [5,6]) <br/>
> bool_idx = (a > 2) # find the elements of a that are bigger than 2 this returns numpy array of the same shape as a. Each slot tells whether that is > 2 <br/>
> print(a[bool_idx]) # Prints "[3 4 5 6]" <br/>
> print(a[a>2]) # Prints "[3 4 5 6]"

##### Datatypes
Every numpy array is a grid of elements of the same type. Numpy provides a large set of numeric datatypes that you can use to construct arrays. Numpy tries to guess a datatype when you create an array, but functions that construct arrays usually also include an optional argument to specify.

x = np.array([1, 2], dtype = np.int64)

##### Array math
- +, -, *, / Ex: <br/>
	x = np.array([1, 2], [3, 4], dtype = np.float64) <br/>
	y = np.array([5, 6], [7, 8], dtype = np.float64) <br/>
    print(x*y)<br/> <br/>
- Unlike matlab, * is elementwise multiplication not matrix multiplication. We use dot function to compute the inner product of vectors, to multiply a vector by a matrix, and to multiply matrices. Dot is both for numpy module and instance method to array objects. <br/>
x = np.array([[1,2],[3,4]]) <br/>
y = np.array([[5,6],[7,8]]) <br/>
v = np.array([9,10]) <br/>
w = np.array([11, 12]) <br/> <br/>
\# Inner product of vectors; both produce 219 <br/>
print(v.dot(w)) <br/>
print(np.dot(v, w)) <br/> <br/>
\# Matrix vector product; both produce the rank 1 array [29, 67] <br/>
print(x.dot(v)) <br/>
print(np.dot(x, v)) <br/> <br/>
\# Matrix matrix product; both produce the rank 2 array <br/>
print(x.dot(y)) <br/>
print(np.dot(x, y))

> x = np.array([1,2],[3,4]) <br/>
print(np.sum(x)) # compute sum of all elements <br/>
print(np.sum(x, axis = 0)) # Compute sum of each column, [4, 6] <br/>
print(np.sum(x, axis = 1)) # Compute sum of each row, [3, 7] <br/>

> x = np.array([1, 2], [3, 4]) <br/>
> print(x.T) # transpose function <br/>
> transpose function doesn't work on rank 1 array.

##### Broadcasting
Broadcasting allows numpy to work with arrays of different shapes. Frequently we have a smaller array and a larger array and we want to use the smaller array multiple times to perform some operation on the larger array.

For example, we want to add a constant vector to each row of a matrix x, storing the result in the matrix y
> x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]) <br/>
> v = np.array([1, 0, 1]) <br/>
> y = np.empty_like(x) # create an empty matrix with the same shape as x <br/>
> for i in range(4): <br/>
	&nbsp; y[i, :] = x[i, :] + v

This works. However, if x is large, computing an explicit loop in Python could be slow. Adding the vector v to each row of the matrix x is equivalent to forming a matrix vv by stacking multiple copies of v vertically, then performing elementwise summation of x and vv.

> x = np.array([1,2,3],[4,5,6],[7,8,9],[10,11,12]) <br/>
> v = np.array([1, 0, 1]) <br/>
> vv = np.tile(v, (4,1)) # stack 4 cpies of v on top of each other
> y = x + vv # elementwise addition

Numpy broadcast allows us perform this computation without actually creating multiple copies of v.

> x = np.array([1,2,3],[4,5,6],[7,8,9],[10,11,12]) <br/>
> v = np.array([1, 0, 1]) <br/>
> y = x + v

1. If the arrays do not have the same rank, prepend the shape of the lower rank array with 1s until both shapes have the same length.
2. The two arrays are said to be compatible in a dimension if they have the same size in the dimension, or if one of the arrays has size 1 in that dimension.
3. In any dimension where one array had size 1 and the other array had size greater than 1, the first array behaves as if it were copied along that dimension.

> x = np.array([[1,2,3], [4,5,6]]) <br/>
> w = np.array([4,5]) <br/>
> \# Add a vector to each column of a matrix <br/>
> \# x has shape (2, 3) and w has shape (2,) <br/>
> \# If we transpose x then it has shape (3,2) and can be broadcast <br/>
> print((x.T + w).T) <br/>
> \# reshape w to be column vector of shape (2, 1) <br/>
> print(x + np.reshape(w, (2, 1))) <br/> <br/>

#### SCIPY
scipy builds on numpy and provides a large number of functions that operate on numpy arrays and are useful for different types of scientific and engineering applications.

##### Image operations
SciPy provides some basic functions to work with images. It has functions to read images from disk into numpy arrays, to write numpy arrays to disk as images, and to resize images.

> from scipy.misc import imread, imsave, imresize <br/>
> img = imread('image.jpg') <br/>
> \# we can tint the image by scaling each of the color channels <br/>
> \# array [1, 0.95, 0.9] has shape (3,) <br/>
> \# numpy broadcasting means that this leaves the red channel unchanged, <br/>
> \# and multiplies the green and blue channels by 0.95 and 0.9 respectively <br/>
> img_tinted = img * [1, 0.95, 0.9] <br/>
> \# Resize the tinted image to be 300 by 300 pixels.
> img_tinted = imresize(img_tinted, (300, 300))
> imsave('assets/cat_tinted.jpg', img_tinted)

##### MATLAB files
The functions scipy.io.loadmat and scipy.io.savemat allow you to read and write MATLAB files.

##### Distance between points
SciPy defines some useful functions for computing distances between sets of points. <br/>
The function scipy.spatial.distance.pdist computes the distance between all pairs of points in a given set: <br/>
> import numpy as np <br/>
> from scipy.spatial.distance import pdist, squareform <br/>
> x = np.array([[0, 1], [1, 0], [2, 0]])
> x = np.array([[0, 1], [1, 0], [2, 0]])

A similar function (scipy.spatial.distance.cdist) computes the distance between all pairs across two sets of point

#### Matplotlib
Matplotlib is a plotting library. In this section give a brief introduction to the matplotlib.pyplot module, which provides a plotting system similar to that of MATLAB.

plot allows you to plot 2D data
subplot

