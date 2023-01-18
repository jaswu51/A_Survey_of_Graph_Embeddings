# papers and algorithms

## Graph Auto Encoder(GAE)
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


## Graph2Vec

### Word2Vec

#### CBOW

<img width="1135" alt="Screenshot 2023-01-18 at 13 32 23" src="https://user-images.githubusercontent.com/91216581/213172527-aaa5149e-8c3e-4902-8e92-1c47674464aa.png">


#### SkipGram

<img width="1135" alt="Screenshot 2023-01-18 at 13 34 06" src="https://user-images.githubusercontent.com/91216581/213172647-dd91e3a9-9d5b-413f-971a-a7cc479be090.png">


### Doc2Vec

In the doc2vec network, the whole document will be also be treated as one of the inputs in the word prediction.
<img width="728" alt="Screenshot 2023-01-18 at 14 00 36" src="https://user-images.githubusercontent.com/91216581/213178215-2cc39ce0-4336-4eca-a83f-11c7e246d516.png">



