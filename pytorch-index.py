#tensor数据的索引与切片操作
import torch
a=torch.rand(4,3,28,28)
print(a)
print(a.shape)
print(a.dim())
#索引与切片操作
print(a[0].shape)
print(a[0,0,1,2])
print(a[:2].shape)
print(a[:2,:1,:,:].shape)
print(a[:2,1:,:,:].shape)
print(a[:2,-3:].shape)
print(a[:,:,0:28:2,0:28:2].shape)
print(a[:,:,::2,::3].shape)
#选择其中某维度的某些索引数据
b=torch.rand(5,3,3)
print(b)
print(b.index_select(0,torch.tensor([1,2,4])))
print(b.index_select(2,torch.arange(2)).shape)
#...操作表示自动判断其中得到维度区间
a=torch.rand(4,3,28,28)
print(a[...,2].shape)
print(a[0,...,::2].shape)
print(a[...].shape)
#msaked_select
x=torch.randn(3,4)
print(x)
mask=x.ge(0.5)          #选出所有元素中大于0.5的数据
print(mask)
print(torch.masked_select(x,mask))  #选出所有元素中大于0.5的数据，并且输出时将其转换为了dim=1的打平tensor数据
#take函数的应用:先将张量数据打平为一个dim=1的张量数据（依次排序下来成为一个数据列），然后按照索引进行取数据
a=torch.tensor([[1,2,3],[4,5,6]])
print(a)
print(a.shape)
print(torch.take(a,torch.tensor([1,2,5])))

#tensor数据的维度变换
#view/reshape操作：不进行额外的记住和存贮就会丢失掉原来的数据的数据和维度顺序信息，而这是非常重要的
a=torch.rand(4,1,28,28)
print(a.view(4,28*28))
b=a.view(4,28*28)
print(b.shape)
#squeeze/unsqueeze挤压和增加维度的操作
a=torch.rand(4,3,28,28)
print(a)
print(a.unsqueeze(0).shape)
print(a.unsqueeze(-1).shape)
print(a.unsqueeze(-4).shape)
a=torch.tensor([1.2,1.3])      #[2]
print(a.unsqueeze(0))          #[1,2]
print(a.unsqueeze(-1))         #[2,1]
a=torch.rand(4,32,28,28)
b=torch.rand(32)   #如果要实现a和数据b的叠加，则需要对于数据b进行维度扩张
print(b.unsqueeze(1).unsqueeze(2).unsqueeze(0).shape)
b=b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(b.shape)
print(b.squeeze().shape)
print(b.squeeze(0).shape)
print(b.squeeze(1).shape)
print(b.squeeze(-1).shape)
#维度的扩张expand（绝对值）/repeat（重复拷贝的次数-相对值，并且由于拷贝操作，原来的数据不能再用，已经改变），只能从1扩张到n，不能从M扩张到N，另外-1表示对该维度保持不变的操作
a=torch.rand(4,32,14,14)
b=torch.rand(32)
b=b.unsqueeze(1).unsqueeze(2).unsqueeze(0)
print(a.shape,b.shape)
print(b.expand(4,32,14,14).shape)
print(b.expand(-1,32,-1,-1).shape) #-1表示对维度保持不变
print(b.repeat(4,32,1,1).shape)
print(b.repeat(4,1,14,14).shape)
#.t()和transpose/permute维度交换操作，需要考虑数据的信息保存，不能出现数据的污染和混乱.contiguous()操作保持存储顺序不变
c=torch.rand(3,4)
print(c)
print(c.t())
a=torch.rand(4,3,32,32)
a1=a.transpose(1,3).contiguous().view(4,32*32*3).view(4,32,32,3).transpose(1,3)
print(a1.shape)
print(torch.all(torch.eq(a,a1)))
a=torch.rand(4,3,28,32)
a1=a.permute(0,2,3,1)
print(a1.shape)
a2=a.contiguous().permute(0,2,3,1)
print(torch.all(torch.eq(a1,a2)))
















