# papers and algorithms

### Graph Auto Encoder(GAE)
Let's first talk about the node-level-embedding GAE structure.

![image](https://user-images.githubusercontent.com/91216581/213168754-b2bb59b0-32b0-4ec2-a57f-7af3fec10cc4.png)



#### Encoder
The encoder is a GCN module, where input is the feature vectors of the node and the adjacency matrix. The output is the embedded vector in the hidden space.


![Alt text](https://img-blog.csdnimg.cn/20200603144512310.png#pic_center)


The structure of the GCN is pretty simple, a parameter matrix mulplication, plus an activation function.


![Alt text](https://img-blog.csdnimg.cn/20200603144824291.png#pic_center)
#### Decoder
![Alt text](https://img-blog.csdnimg.cn/20200603145649540.png#pic_center)

The decoder tries to rebuild the ajacency matrix via the hidden vectors' dot product. So the loss function is pretty intuitive as well, the difference of the rebuilt ajacency matrix and the real ajacency matrix. 


![Alt text](https://img-blog.csdnimg.cn/20200603145944227.png#pic_center)
