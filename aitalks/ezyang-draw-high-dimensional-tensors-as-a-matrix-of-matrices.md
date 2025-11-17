Draw high dimensional tensors as a matrix of matrices
-----------------------------------------------------

I have recently needed to draw the contents of high-dimensional (e.g., 4D and up) tensors where it is important to ensure that is clear how to identify each of the dimensions in the representation. Common strategies I've seen people do in this situation include printing a giant list 2D slices (what the default PyTorch printer will do) or flattening the Tensor in some way back down to a 2D tensor. However, if you have a lot of horizontal space, there is a strategy that I like that makes it easy to identify all the axes of the higher dimensional tensor: draw it as a matrix of matrices.

Here are some examples, including the easy up-to-2D cases for completeness.

0D: torch.arange(1).view()

0

1D: torch.arange(2)

0  1

2D: torch.arange(4).view(2, 2 )

0  1
2  3

3D: torch.arange(8).view(2, 2, 2)

0  1    4  5
2  3    6  7

4D: torch.arange(16).view(2, 2, 2, 2)

 0  1    4  5
 2  3    6  7

 8  9   12 13
10 11   14 15

5D: torch.arange(32).view(2, 2, 2, 2, 2):

 0  1    4  5  :  16 17   20 21
 2  3    6  7  :  18 19   22 23
               :
 8  9   12 13  :  24 25   28 29
10 11   14 15  :  26 27   30 31

The idea is that every time you add a new dimension, you alternate between stacking the lower dimension matrices horizontally and vertically. You always stack horizontally _before_ stacking vertically, to follow the standard row-major convention for printing in the 2D case. Dimensions always proceed along the x and y axis, but the higher dimensions (smaller dim numbers) involve skipping over blocks. For example, a "row" on dim 3 in the 4D tensor is \[0, 1\] but the "row" on dim 1 is \[0, 4\] (we skip over to the next block.) The fractal nature of the construction means we can keep repeating the process for as many dimensions as we like.

In fact, for the special case when every size in the tensor is 2, the generated sequence of indices form a [Morton curve](https://en.wikipedia.org/wiki/Z-order_curve). But I don't call it that, since I couldn't find a popular name for the variation of the Morton curve where the radix of each digit in the coordinate representation can vary.

**Knowledge check.** For the 4D tensor of size (2, 2, 2, 2) arranged in this way, draw the line(s) that would split the tensor into the pieces that torch.split(x, 1, dim), for each possible dimension 0, 1, 2 and 3. Answer under the fold.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

.

dim=0

>>> \[x.reshape(-1) for x in torch.arange(16).view(2,2,2,2).split(1,dim=0)\]
\[tensor(\[0, 1, 2, 3, 4, 5, 6, 7\]), tensor(\[ 8, 9, 10, 11, 12, 13, 14, 15\])\]

     0  1    4  5
     2  3    6  7
   ----------------
     8  9   12 13
    10 11   14 15


dim=1

>>> \[x.reshape(-1) for x in torch.arange(16).view(2,2,2,2).split(1,dim=1)\]
\[tensor(\[ 0, 1, 2, 3, 8, 9, 10, 11\]), tensor(\[ 4, 5, 6, 7, 12, 13, 14, 15\])\]

     0  1 |  4  5
     2  3 |  6  7
          |
     8  9 | 12 13
    10 11 | 14 15

dim=2

>>> \[x.reshape(-1) for x in torch.arange(16).view(2,2,2,2).split(1,dim=2)\]
\[tensor(\[ 0, 1, 4, 5, 8, 9, 12, 13\]), tensor(\[ 2, 3, 6, 7, 10, 11, 14, 15\])\]

     0  1    4  5
   ------- -------
     2  3    6  7

     8  9   12 13
   ------- -------
    10 11   14 15

dim=3

>>> \[x.reshape(-1) for x in torch.arange(16).view(2,2,2,2).split(1,dim=3)\]
\[tensor(\[ 0, 2, 4, 6, 8, 10, 12, 14\]), tensor(\[ 1, 3, 5, 7, 9, 11, 13, 15\])\]

     0 |  1    4 |  5
     2 |  3    6 |  7

     8 |  9   12 | 13
    10 | 11   14 | 15