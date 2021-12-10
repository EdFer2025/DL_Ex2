if __name__ != "__main__":
    from Layers.Base import BaseLayer
else:
    from Base import BaseLayer
import numpy as np 

# python .\NeuralNetworkTests.py TestPooling
class Pooling(BaseLayer):
    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.batch_size = None
        self.num_channels = None
        self.input_shape = None
        self.pool_positions = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        """print(input_tensor.shape)
        print(self.pooling_shape)
        print(self.stride_shape)"""
        self.batch_size = input_tensor.shape[0]
        self.num_channels = input_tensor.shape[1]
        self.pool_positions = {}
        
        pooled_vector_shape = (self.batch_size, 
                                self.num_channels, 
                                input_tensor.shape[2] - self.pooling_shape[0] + 1, 
                                input_tensor.shape[3] - self.pooling_shape[1] + 1)
        pooled_vector = np.zeros(shape=pooled_vector_shape)
        for b in range(self.batch_size):
            for c in range(self.num_channels):
                data = input_tensor[b, c]
                for i in range(self.pooling_shape[0], data.shape[0] + 1):
                    for j in range(self.pooling_shape[1], data.shape[1] + 1):
                        sec = data[i-self.pooling_shape[0]:i, j-self.pooling_shape[1]:j]
                        position = np.argmax(sec)
                        #print(position)
                        pos_x = position//self.pooling_shape[0] + i-self.pooling_shape[0]
                        pos_y = position%self.pooling_shape[1] + j-self.pooling_shape[1]
                        self.pool_positions[(i-self.pooling_shape[0], j-self.pooling_shape[1])] = pos_x, pos_y
                        # print(pos_x, pos_y)
                        #pooled_vector[b, c, i-self.pooling_shape[0], j-self.pooling_shape[1]] = np.max(sec)
                        pooled_vector[b, c, i-self.pooling_shape[0], j-self.pooling_shape[1]] = data[pos_x, pos_y]
        # print(self.pool_positions)                

        # For striding
        idxs = []
        input_dimensions = pooled_vector.shape[2:]
        for i in range(len(input_dimensions)):
            ax = input_dimensions[i]
            idx = list(range(0, ax, self.stride_shape[i]))
            idxs.append(idx)
        # print(idxs)
        

        sampled_batch_output = np.zeros(shape=(self.batch_size, self.num_channels, len(idxs[0]), len(idxs[1])))
        for b in range(self.batch_size):
            for k in range(self.num_channels):
                for i in range(len(idxs[0])):
                    for j in range(len(idxs[1])):
                        sampled_batch_output[b, k, i, j] = pooled_vector[b, k, idxs[0][i], idxs[1][j]]
                        self.pool_positions[(i,j)] = self.pool_positions[idxs[0][i], idxs[1][j]]
        # print(self.pool_positions)
        return sampled_batch_output
    
    def backward(self, error_tensor):
        error_dimensions = error_tensor.shape[2:]
        print(error_dimensions)

        input_gradient = np.zeros(shape=self.input_shape)
        for b in range(self.batch_size):
            for c in range(self.num_channels):
                for i in range(error_dimensions[0]):
                    for j in range(error_dimensions[1]):
                        x, y = self.pool_positions[i, j]
                        input_gradient[b, c, x, y] += error_tensor[b, c, i,j]
                pass
        return input_gradient

def _test_forward():
    input_vector = np.array([
        
        [
            [
                [9,8,0,5],
                [3,5,1,1],
                [1,1,6,3],
                [5,2,6,3]
            ]
        ]
        
    ])
    print(input_vector.shape)
    print(input_vector)
    strides = (1,1)
    p = Pooling(strides, (2,2))
    f = p.forward(input_vector)
    print(f)

    error_tensor = None
    if strides == (1,1):
        error_tensor = np.array([
            [
                [
                    [6,4,3],
                    [2,5,4],
                    [7,1,2]
                ]
            ]
        ])
    elif strides == (2,2):
        error_tensor = np.array([
            [
                [
                    [6,3],
                    [7,2]
                ]
            ]
        ])

    input_gradient = p.backward(error_tensor)
    print(input_gradient)


if __name__=="__main__":
    from Base import BaseLayer
    _test_forward()