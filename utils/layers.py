from tensorflow.keras.layers import Layer, Dense


class FullyConnected(Layer):
    def __init__(self, n_fc, hidden_phi, out_size,  final_activation, name, kernel_reg, kernel_init, activation='elu',
                 bias_initializer=None, **kwargs):
        super(FullyConnected, self).__init__(name=name, **kwargs)
        self.Layers = []
        for i in range(n_fc-1):
            self.Layers.append(Dense(units=hidden_phi, activation=activation, kernel_initializer=kernel_init,
                                     bias_initializer=bias_initializer,
                                     kernel_regularizer=kernel_reg, name=name + str(i)))
        self.Layers.append(Dense(units=out_size, activation=final_activation, name=name + 'out'))

    def call(self, x):
        for layer in self.Layers:
            x = layer(x)
        return x
